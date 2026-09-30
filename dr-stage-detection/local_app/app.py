"""
Diabetic Retinopathy Stage Detection - desktop app (Gradio), Version 5 interface
================================================================================
Upload a retinal fundus photograph and the app shows:
  * the DR stage (0-4) predicted by the CNN, with the probability of every stage
  * a DR / No DR screening decision and a suggested clinical action
  * the image the model actually sees and a Grad-CAM heat map of where it looked
  * the model's measured performance on the held-out test set

Files needed in the same folder (all produced by the Kaggle notebook, in dr_outputs.zip):
  dr_config.json and every model it lists             (required)
      single model : dr_model.keras
      ensemble     : dr_model.keras, dr_model_member1.keras, dr_model_member2.keras (Version 4)
  results.json, classification_report.csv, figures/, examples/   (optional, shown in the app)

Version 4 uses an ensemble of three CNNs (EfficientNetB0 at 300 px, EfficientNetB3 and EfficientNetB4 at
380 px). Each model gets the image at its own input size and the stage probabilities are averaged.

The interface matches the hosted website: a sidebar with ten pages (overview, analyse, dataset,
pipeline, models, training, results, explainability, ethics, about). Page text and styles are in
ui/pages.html and ui/style.css; figures are read from the figures/ folder.

Run:  python app.py      (or double-click start_app.bat on Windows)
"""
import glob
import html
import inspect
import json
import os
import re
import time
from pathlib import Path

import cv2
import gradio as gr
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------------------------------
# 1. Load the trained model and its settings
# ---------------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = json.load(open(os.path.join(HERE, "dr_config.json")))
IMG_SIZE = CONFIG["img_size"]
MODE = CONFIG["preprocess_mode"]                    # "raw", "clahe_unsharp" or "ben_graham"
USE_TTA = CONFIG.get("tta", False)                  # average predictions over flipped views
DECISION = CONFIG.get("decision", "argmax")         # "argmax" or "thresholds"
THRESHOLDS = np.array(CONFIG.get("thresholds", [0.5, 1.5, 2.5, 3.5]))
BACKBONE = CONFIG.get("backbone", "CNN")

# Each entry: (display name, Keras model, input size in pixels)
ENSEMBLE = CONFIG.get("ensemble")
if ENSEMBLE:                                         # Version 4: several models, possibly different sizes
    USE_TTA = ENSEMBLE.get("tta", USE_TTA)
    MODELS = []
    for member in ENSEMBLE["members"]:
        print(f"Loading {member['name']} ({member['file']}) ...")
        MODELS.append((member["name"], keras.models.load_model(os.path.join(HERE, member["file"])),
                       int(member.get("img_size", IMG_SIZE))))
    MODEL_LABEL = f"Ensemble of {len(MODELS)} CNNs"
    SIZES_LABEL = " / ".join(sorted({f"{s}px" for _, _, s in MODELS}))
else:                                                # Version 1-2: one model
    MODELS = [(BACKBONE, keras.models.load_model(os.path.join(HERE, "dr_model.keras")), IMG_SIZE)]
    MODEL_LABEL = BACKBONE
    SIZES_LABEL = f"{IMG_SIZE}px"

RESULTS = None
if os.path.exists(os.path.join(HERE, "results.json")):
    RESULTS = json.load(open(os.path.join(HERE, "results.json")))

STAGES = [  # name, short description, suggested action, colour
    ("No DR", "No visible signs of diabetic retinopathy.",
     "Routine screening again in 12 months. Keep blood sugar, blood pressure and cholesterol under control.",
     "#2f9e6f"),
    ("Mild NPDR", "Microaneurysms only: tiny bulges in the retinal blood vessels.",
     "Re-screen in 6 to 12 months and review diabetes control with the patient's doctor.",
     "#8fae2b"),
    ("Moderate NPDR", "More microaneurysms, dot/blot haemorrhages and hard exudates.",
     "Refer to an ophthalmologist within about 3 months.",
     "#d99a14"),
    ("Severe NPDR", "Many haemorrhages in all quadrants, venous beading or abnormal vessels (4-2-1 rule).",
     "Urgent referral to an ophthalmologist within weeks: high risk of progressing to proliferative DR.",
     "#e0662a"),
    ("Proliferative DR", "New fragile blood vessels grow on the retina and can bleed or detach it.",
     "Urgent referral: treatment such as laser photocoagulation or anti-VEGF injections may be needed.",
     "#cf3b3b"),
]


# ---------------------------------------------------------------------------------------------
# 2. Preprocessing: identical to the training notebook
# ---------------------------------------------------------------------------------------------
def crop_black_borders(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=0)


def apply_clahe(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def unsharp_mask(img, sigma=2.0, amount=0.5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def ben_graham(img, sigma=None):
    sigma = sigma or img.shape[0] / 30
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


def circular_mask(img, scale=0.96):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * scale), 1, -1)
    return img * mask[..., None]


def preprocess(img_rgb, size=IMG_SIZE):
    img = pad_to_square(crop_black_borders(img_rgb))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if MODE == "raw":
        return img
    img = cv2.medianBlur(img, 3)
    img = unsharp_mask(apply_clahe(img)) if MODE == "clahe_unsharp" else ben_graham(img)
    return circular_mask(img)


# ---------------------------------------------------------------------------------------------
# 3. Grad-CAM and prediction: same method as the notebook
# ---------------------------------------------------------------------------------------------
def cam_for_model(model, img_uint8, cls):
    # Grad-CAM for one model: gradients of the chosen stage w.r.t. the backbone's last feature map
    base_layer = next(l for l in model.layers if isinstance(l, keras.Model))
    pos = model.layers.index(base_layer)
    x = tf.convert_to_tensor(img_uint8[None].astype("float32"))
    with tf.GradientTape() as tape:
        h = x
        for layer in model.layers[1:pos]:
            h = layer(h)
        conv = base_layer(h, training=False)
        tape.watch(conv)
        h = conv
        for layer in model.layers[pos + 1:]:
            h = layer(h, training=False)
        score = h[:, cls]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    return cam / (cam.max() + 1e-8)


def grad_cam(inputs, cls, show_img):
    # Ensemble Grad-CAM: each model's map (for the ensemble's stage) resized to the display size and averaged
    size = (show_img.shape[1], show_img.shape[0])
    cams = [cv2.resize(cam_for_model(m, inputs[s], cls), size) for _, m, s in MODELS]
    cam = np.mean(cams, axis=0)
    cam = cam / (cam.max() + 1e-8)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(show_img, 0.55, heat, 0.45, 0)


def stage_probabilities(inputs):
    # inputs: {size: preprocessed image}. Returns the averaged probabilities and each model's own probabilities
    member_probs = []
    for _, model, size in MODELS:
        pre = inputs[size]
        batch = np.stack([pre, pre[:, ::-1], pre[::-1], pre[::-1, ::-1]]) if USE_TTA else pre[None]
        member_probs.append(model.predict(batch.astype("float32"), verbose=0).mean(0))
    return np.mean(member_probs, axis=0), member_probs


def decide(probs):
    if DECISION == "thresholds":
        grade = float(probs @ np.arange(len(probs)))          # expected grade 0-4
        return int(np.digitize(grade, THRESHOLDS))
    return int(np.argmax(probs))




# ---------------------------------------------------------------------------------------------
# 4. HTML building blocks for the result card (same markup and style as the website)
# ---------------------------------------------------------------------------------------------
def placeholder_card():
    return """<div class="rc rc-empty"><div class="rc-empty-icon">&#9673;</div>
      <div class="rc-empty-title">No image analysed yet</div>
      <div class="rc-empty-text">Upload a colour fundus photograph or click one of the test images, then press
      <b>Analyse image</b>. The stage, the probability of each stage and a heat map of where the models looked
      will appear here.</div></div>"""


def error_card(message):
    return f"""<div class="rc rc-empty"><div class="rc-empty-title">Could not analyse this image</div>
               <div class="rc-empty-text">{html.escape(message)}</div></div>"""


def display_name(name):
    # "B3 380px (seed 11)" -> "EfficientNetB3"
    m = re.search(r"\bB(\d)\b", name)
    return f"EfficientNetB{m.group(1)}" if m else name


def members_table(member_probs, stage):
    # How each model in the ensemble voted, so the averaging is visible
    if len(MODELS) < 2:
        return ""
    rows = "".join(
        f'<tr><td>{html.escape(display_name(n))}</td><td>{s}px</td><td>{int(np.argmax(p))} &middot; {STAGES[int(np.argmax(p))][0]}</td>'
        f'<td>{p[stage] * 100:.1f}%</td></tr>'
        for (n, _, s), p in zip(MODELS, member_probs))
    return f"""<div class="rc-sec">How each model in the ensemble voted</div>
      <div class="tbl-wrap members"><table class="tbl"><thead><tr><th>Model</th><th>Input</th><th>Its stage</th>
      <th>P(stage {stage})</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def result_card(probs, stage, seconds, member_probs=()):
    name, desc, action, colour = STAGES[stage]
    p_dr = float(1 - probs[0])
    dr = p_dr >= 0.5
    scale = "".join(f'<div class="seg{" on" if i == stage else ""}" style="--c:{s[3]}"><span class="seg-n">{i}</span>'
                    f'<span class="seg-t">{s[0]}</span></div>' for i, s in enumerate(STAGES))
    bars = "".join(f'<div class="bar-row{" on" if i == stage else ""}"><span>{i} &middot; {s[0]}</span>'
                   f'<span class="bar-track"><span class="bar-fill" style="width:{probs[i] * 100:.1f}%;--c:{s[3]}"></span></span>'
                   f'<span class="bar-pct">{probs[i] * 100:.1f}%</span></div>' for i, s in enumerate(STAGES))
    return f"""<div class="rc" style="--c:{colour}">
      <div class="rc-head"><span class="rc-pill">{'DR' if dr else 'NO DR'}</span>
        <span class="rc-verdict">{'Diabetic retinopathy detected' if dr else 'No diabetic retinopathy detected'}</span></div>
      <div class="rc-stage">Stage {stage} &middot; {name}</div>
      <div class="rc-desc">{desc}</div>
      <div class="kpis">
        <div><span class="k-v">{probs[stage] * 100:.1f}%</span><span class="k-l">confidence in this stage</span></div>
        <div><span class="k-v">{p_dr * 100:.1f}%</span><span class="k-l">probability of any DR</span></div>
        <div><span class="k-v">{seconds:.1f} s</span><span class="k-l">analysis time on this computer</span></div>
      </div>
      <div class="rc-sec">Severity scale</div><div class="scale">{scale}</div>
      <div class="rc-sec">Probability of each stage</div><div class="bars">{bars}</div>
      {members_table(member_probs, stage)}
      <div class="rc-action"><b>Suggested next step:</b> {action}</div>
      <div class="rc-foot">Research prototype for coursework. It is not a medical device and must not be used for
      diagnosis. Model: {MODEL_LABEL}, input {SIZES_LABEL}, preprocessing {MODE}, TTA {'on' if USE_TTA else 'off'},
      decision {DECISION}.</div>
    </div>"""


# ---------------------------------------------------------------------------------------------
# 5. Main callback
# ---------------------------------------------------------------------------------------------
def analyse(image):
    if image is None:
        return placeholder_card(), None, None
    try:
        t0 = time.time()
        img = image[..., :3].astype(np.uint8) if image.ndim == 3 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cropped = pad_to_square(crop_black_borders(img))
        inputs = {s: preprocess(img, s) for s in {size for _, _, size in MODELS}}   # one image per input size
        probs, member_probs = stage_probabilities(inputs)
        stage = decide(probs)
        show = inputs[max(inputs)]                                   # largest input is shown and overlaid
        overlay = grad_cam(inputs, stage, show)
        return result_card(probs, stage, time.time() - t0, member_probs), (show, overlay), cropped
    except Exception as exc:                        # show the problem in the app instead of crashing
        return error_card(str(exc)), None, None


def clear():
    return None, placeholder_card(), None, None


# ---------------------------------------------------------------------------------------------
# 6. Information pages (shared text with the website, in ui/pages.html)
# ---------------------------------------------------------------------------------------------
def pct(v):
    return f"{v * 100:.1f}%"


FIG_DIR = os.path.join(HERE, "figures")
UI_DIR = os.path.join(HERE, "ui")


def fig_url(name):
    # Gradio serves files from allowed folders at /gradio_api/file=<absolute path>
    return "/gradio_api/file=" + Path(FIG_DIR, name).as_posix()


def facts_html():
    if not RESULTS:
        return ""
    m, b = RESULTS["test_stage_metrics"], RESULTS["test_binary_metrics"]
    facts = [(pct(m["accuracy"]), "stage accuracy on 550 unseen test images"),
             (f"{m['qwk']:.3f}", "quadratic weighted kappa, the official APTOS metric"),
             (pct(b["sensitivity (recall)"]), "of eyes with retinopathy correctly flagged"),
             (str(len(MODELS)), "CNNs in the ensemble, running on this computer")]
    return '<div class="facts">' + "".join(f"<div><b>{v}</b><span>{l}</span></div>" for v, l in facts) + "</div>"


def metrics_html():
    if not RESULTS:
        return "<p>Put <code>results.json</code> from the notebook output next to <code>app.py</code> to show the test results.</p>"
    m, b = RESULTS["test_stage_metrics"], RESULTS["test_binary_metrics"]
    tiles = [("Stage accuracy", pct(m["accuracy"]), "exact stage correct"),
             ("Macro F1", f"{m['macro_f1']:.3f}", "average over the 5 stages"),
             ("QWK", f"{m['qwk']:.3f}", "agreement with graders (official APTOS metric)"),
             ("Sensitivity", pct(b["sensitivity (recall)"]), "DR cases correctly flagged"),
             ("Specificity", pct(b["specificity"]), "healthy eyes correctly cleared"),
             ("ROC AUC", f"{b['roc_auc']:.3f}", "DR vs No DR separation")]
    out = '<div class="tiles">' + "".join(f'<div class="tile"><div class="t-l">{t}</div><div class="t-v">{v}</div>'
                                          f'<div class="t-s">{s}</div></div>' for t, v, s in tiles) + "</div>"
    csv_path = os.path.join(HERE, "classification_report.csv")
    if os.path.exists(csv_path):
        rep = pd.read_csv(csv_path, index_col=0)
        rows = "".join(
            f'<tr><td><span class="dot" style="--c:{STAGES[i][3]}"></span>{i} &middot; {STAGES[i][0]}</td>'
            f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1-score']:.3f}</td><td>{int(r['support'])}</td></tr>"
            for i, (_, r) in enumerate(rep.iloc[:5].iterrows()))
        out += f"""<div class="sub-h">Per-stage results on the test set</div><div class="tbl-wrap"><table class="tbl">
          <thead><tr><th>Stage</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Images</th></tr></thead>
          <tbody>{rows}</tbody></table></div>"""
    return out


def load_pages():
    text = open(os.path.join(UI_DIR, "pages.html"), encoding="utf-8").read()
    pages = {}
    for chunk in text.split("<!--PAGE ")[1:]:
        name, body = chunk.split("-->", 1)
        body = re.sub(r'src="\{FIG\}/([^"]+)"', lambda m: f'src="{fig_url(m.group(1))}"', body)
        pages[name.strip()] = body.replace("{FACTS}", facts_html()).replace("{METRICS}", metrics_html())
    return pages


PAGES = load_pages()
NAV = [("Use", [("overview", "Overview"), ("analyse", "Analyse an image")]),
       ("Project", [("dataset", "Dataset"), ("pipeline", "Pipeline"), ("models", "Models"), ("training", "Training")]),
       ("Evidence", [("results", "Results"), ("explain", "Explainability")]),
       ("Context", [("ethics", "Ethics and limits"), ("about", "About and links")])]
ORDER = [key for _, items in NAV for key, _ in items]

BRAND = """<div class="brand"><svg viewBox="0 0 40 40" aria-hidden="true"><circle cx="20" cy="20" r="18" fill="var(--retina)"/>
  <circle cx="27" cy="17" r="5" fill="var(--disc)"/><path d="M6 22c5-5 9 4 14 0s8-6 14-1" stroke="var(--vessel)" stroke-width="2"
  fill="none" stroke-linecap="round"/><path d="M10 12c4 2 7 7 10 5" stroke="var(--vessel)" stroke-width="1.6" fill="none"
  stroke-linecap="round"/></svg><span><b>DR Stage</b><br>Detection</span></div>"""

# Links inside the pages (for example "Analyse an image") click the matching sidebar button
HEAD = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap">
<script>
document.addEventListener("click", function (e) {
  var a = e.target.closest && e.target.closest('a[href^="#"]');
  if (!a) return;
  var btn = document.getElementById("nav-" + a.getAttribute("href").slice(1));
  if (btn) { e.preventDefault(); btn.click(); window.scrollTo(0, 0); }
});
</script>"""

THEME_JS = """() => { document.body.classList.toggle('dark'); document.documentElement.classList.toggle('dark'); }"""

CSS = open(os.path.join(UI_DIR, "style.css"), encoding="utf-8").read()
THEME = gr.themes.Base(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Public Sans"), "Segoe UI", "system-ui", "sans-serif"],
                       font_mono=["Consolas", "monospace"]).set(
    button_primary_background_fill="#28418c", button_primary_background_fill_hover="#1f3373",
    button_primary_text_color="#ffffff", body_background_fill="#f2f5f9", body_background_fill_dark="#0d1420")

# ---------------------------------------------------------------------------------------------
# 7. Layout: sidebar navigation + ten pages
# ---------------------------------------------------------------------------------------------
examples_dir = os.path.join(HERE, "examples")
example_files = sorted(glob.glob(os.path.join(examples_dir, "*.png")) + glob.glob(os.path.join(examples_dir, "*.jp*g")))


def example_label(path):
    name = os.path.basename(path)
    return f"Stage {name[5]}" if name.startswith("stage") and name[5:6].isdigit() else name


with gr.Blocks(title="DR Stage Detection") as demo:
    buttons, views = {}, {}
    with gr.Sidebar(open=True, width=270):
        gr.HTML(BRAND, container=False, padding=False)
        for group, items in NAV:
            gr.HTML(f'<div class="nav-group">{group}</div>', container=False, padding=False)
            for key, label in items:
                buttons[key] = gr.Button(label, elem_id=f"nav-{key}", size="sm",
                                         elem_classes=["navbtn"] + (["active"] if key == "overview" else []))
        gr.HTML(f'<div class="side-state"><i></i>{len(MODELS)} models ready</div>', container=False, padding=False)
        theme_btn = gr.Button("Switch light / dark", size="sm", elem_classes=["navbtn"])

    for key in ORDER:
        with gr.Column(visible=(key == "overview")) as views[key]:
            if key != "analyse":
                gr.HTML(PAGES[key], container=False)
                continue
            # ---- the analyse page is built from Gradio components ----
            gr.HTML('<div class="an-title"><h1>Analyse an image</h1><p>Choose a colour fundus photograph. The three '
                    'models run on this computer and the result appears on the right.</p></div>', container=False)
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    inp = gr.Image(type="numpy", label="Fundus photograph", height=320, sources=["upload", "clipboard"])
                    with gr.Row():
                        btn = gr.Button("Analyse image", variant="primary", elem_id="analyse-btn", scale=3)
                        clr = gr.Button("Clear", variant="secondary", elem_id="clear-btn", scale=1)
                    if example_files:
                        gr.HTML('<div class="an-label">Or click an unseen test image (true stage shown)</div>', container=False, padding=False)
                        gallery = gr.Gallery(value=[(p, example_label(p)) for p in example_files], columns=5, rows=2,
                                             height=240, allow_preview=False, show_label=False, object_fit="cover",
                                             elem_id="examples")
                with gr.Column(scale=6):
                    result = gr.HTML(placeholder_card(), container=False)
            gr.HTML('<h2 class="h2">Where the models looked</h2>', container=False)
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    lens = gr.ImageSlider(label="Model input  |  Grad-CAM heat map (drag the handle)", height=440,
                                          interactive=False)
                    gr.HTML('<div class="slider-note">Photo<span class="jet"></span>heat map: low to high influence. '
                            'The heat map is the average of the three models.</div>', container=False, padding=False)
                with gr.Column(scale=2):
                    orig_out = gr.Image(label="Original (cropped)", height=300, interactive=False)
    gr.HTML('<div class="foot">Research prototype for coursework. Not for clinical use.</div>', container=False)

    # ---- navigation: show one page, highlight its button ----
    def go(target):
        return ([gr.update(visible=(k == target)) for k in ORDER] +
                [gr.update(elem_classes=["navbtn"] + (["active"] if k == target else [])) for k in ORDER])
    for key in ORDER:
        buttons[key].click(lambda k=key: go(k), outputs=[views[k] for k in ORDER] + [buttons[k] for k in ORDER])
    theme_btn.click(None, js=THEME_JS)

    # ---- analyse actions ----
    btn.click(analyse, inputs=inp, outputs=[result, lens, orig_out])
    if example_files:
        def pick(evt: gr.SelectData):
            return cv2.cvtColor(cv2.imread(example_files[evt.index]), cv2.COLOR_BGR2RGB)
        gallery.select(pick, outputs=inp).then(analyse, inputs=inp, outputs=[result, lens, orig_out])
    inp.upload(lambda: (placeholder_card(), None, None), outputs=[result, lens, orig_out])
    clr.click(clear, outputs=[inp, result, lens, orig_out])


if __name__ == "__main__":
    # theme/css/head are launch() options in Gradio 6 and Blocks() options in older versions
    launch_params = inspect.signature(demo.launch).parameters
    extra = {k: v for k, v in {"theme": THEME, "css": CSS, "head": HEAD}.items() if k in launch_params}
    if len(extra) < 3:
        demo.theme, demo.css, demo.head = THEME, CSS, HEAD
    on_server = bool(os.environ.get("SPACE_ID"))        # set automatically on Hugging Face Spaces
    demo.launch(inbrowser=not on_server, allowed_paths=[FIG_DIR], **extra)
