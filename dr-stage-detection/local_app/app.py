"""
Diabetic Retinopathy Stage Detection - prototype web app (Gradio)
=================================================================
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

Run:  python app.py      (or double-click start_app.bat on Windows)
"""
import glob
import html
import inspect
import json
import os
import time

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
# 4. HTML building blocks for the result card
# ---------------------------------------------------------------------------------------------
def placeholder_card():
    return """
    <div class="rc rc-empty">
      <div class="rc-empty-icon">&#9673;</div>
      <div class="rc-empty-title">No image analysed yet</div>
      <div class="rc-empty-text">Upload a retinal fundus photograph or pick one of the example images,
      then press <b>Analyse image</b>. The result, the stage probabilities and a heat map of where the
      model looked will appear here.</div>
    </div>"""


def error_card(message):
    return f"""<div class="rc rc-error"><div class="rc-empty-title">Could not analyse this image</div>
               <div class="rc-empty-text">{html.escape(message)}</div></div>"""


def members_table(member_probs, stage):
    # Shows how each model in the ensemble voted, so the averaging is visible
    if len(MODELS) < 2:
        return ""
    rows = "".join(
        f'<tr><td>{html.escape(n)}</td><td>{s}px</td><td>{int(np.argmax(p))} &middot; {STAGES[int(np.argmax(p))][0]}</td>'
        f'<td>{p[stage] * 100:.1f}%</td></tr>'
        for (n, _, s), p in zip(MODELS, member_probs))
    return f"""<div class="rc-sec">How each model in the ensemble voted</div>
      <div class="tbl-wrap"><table class="tbl members"><thead><tr><th>Model</th><th>Input</th><th>Its stage</th>
      <th>P(stage {stage})</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def result_card(probs, stage, seconds, member_probs=()):
    name, desc, action, colour = STAGES[stage]
    p_dr = float(1 - probs[0])
    dr = p_dr >= 0.5
    verdict = "Diabetic retinopathy detected" if dr else "No diabetic retinopathy detected"

    scale = "".join(
        f'<div class="seg{" seg-on" if i == stage else ""}" style="--c:{s[3]}">'
        f'<span class="seg-n">{i}</span><span class="seg-t">{s[0]}</span></div>'
        for i, s in enumerate(STAGES))

    bars = "".join(
        f'<div class="bar-row{" bar-on" if i == stage else ""}">'
        f'<span class="bar-label">{i} &middot; {s[0]}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{probs[i] * 100:.1f}%;--c:{s[3]}"></span></span>'
        f'<span class="bar-pct">{probs[i] * 100:.1f}%</span></div>'
        for i, s in enumerate(STAGES))

    return f"""
    <div class="rc" style="--c:{colour}">
      <div class="rc-head">
        <span class="rc-pill">{'DR' if dr else 'NO DR'}</span>
        <span class="rc-verdict">{verdict}</span>
      </div>
      <div class="rc-stage">Stage {stage} &middot; {name}</div>
      <div class="rc-desc">{desc}</div>
      <div class="rc-kpis">
        <div><span class="k-v">{probs[stage] * 100:.1f}%</span><span class="k-l">confidence in this stage</span></div>
        <div><span class="k-v">{p_dr * 100:.1f}%</span><span class="k-l">probability of any DR</span></div>
        <div><span class="k-v">{seconds:.1f} s</span><span class="k-l">analysis time</span></div>
      </div>
      <div class="rc-sec">Severity scale</div>
      <div class="scale">{scale}</div>
      <div class="rc-sec">Probability of each stage</div>
      <div class="bars">{bars}</div>
      {members_table(member_probs, stage)}
      <div class="rc-action"><b>Suggested next step:</b> {action}</div>
      <div class="rc-foot">Research prototype for coursework. It is not a medical device and must not be used
      for diagnosis. Model: {MODEL_LABEL} &middot; input {SIZES_LABEL} &middot; preprocessing: {MODE}
      &middot; TTA: {'on' if USE_TTA else 'off'} &middot; decision: {DECISION}</div>
    </div>"""


# ---------------------------------------------------------------------------------------------
# 5. Main callback
# ---------------------------------------------------------------------------------------------
def analyse(image):
    if image is None:
        return placeholder_card(), None, None, None
    try:
        t0 = time.time()
        img = image[..., :3].astype(np.uint8) if image.ndim == 3 else cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        cropped = pad_to_square(crop_black_borders(img))
        inputs = {s: preprocess(img, s) for s in {size for _, _, size in MODELS}}   # one image per input size
        probs, member_probs = stage_probabilities(inputs)
        stage = decide(probs)
        show = inputs[max(inputs)]                                   # largest input is shown and overlaid
        overlay = grad_cam(inputs, stage, show)
        return result_card(probs, stage, time.time() - t0, member_probs), cropped, show, overlay
    except Exception as exc:                        # show the problem in the app instead of crashing
        return error_card(str(exc)), None, None, None


def clear():
    return None, placeholder_card(), None, None, None


# ---------------------------------------------------------------------------------------------
# 6. Content for the information tabs
# ---------------------------------------------------------------------------------------------
def pct(v):
    return f"{v * 100:.1f}%"


def header_html():
    chips = [f"Model: {MODEL_LABEL}", f"Input: {SIZES_LABEL}", "Dataset: APTOS 2019 (3,662 images)"]
    if RESULTS:
        m = RESULTS["test_stage_metrics"]
        chips += [f"Test accuracy: {pct(m['accuracy'])}", f"QWK: {m['qwk']:.3f}"]
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    return f"""
    <div class="hero">
      <div class="hero-eyebrow">Computer Vision &middot; CNN transfer learning</div>
      <h1>Diabetic Retinopathy Stage Detection</h1>
      <p>Grades a retinal fundus photograph on the international 0&ndash;4 diabetic retinopathy scale and shows
      where the model looked when it made its decision.</p>
      <div class="chips">{chip_html}</div>
    </div>"""


def metrics_html():
    if not RESULTS:
        return "<p>Put <code>results.json</code> from the notebook output next to <code>app.py</code> to show the test results here.</p>"
    m, b = RESULTS["test_stage_metrics"], RESULTS["test_binary_metrics"]
    tiles = [
        ("Stage accuracy", pct(m["accuracy"]), "exact stage correct"),
        ("Macro F1", f"{m['macro_f1']:.3f}", "average over the 5 stages"),
        ("QWK", f"{m['qwk']:.3f}", "agreement with graders (official APTOS metric)"),
        ("Sensitivity", pct(b["sensitivity (recall)"]), "DR cases correctly flagged"),
        ("Specificity", pct(b["specificity"]), "healthy eyes correctly cleared"),
        ("ROC AUC", f"{b['roc_auc']:.3f}", "DR vs No DR separation"),
    ]
    tile_html = "".join(f'<div class="tile"><div class="t-l">{t}</div><div class="t-v">{v}</div>'
                        f'<div class="t-s">{s}</div></div>' for t, v, s in tiles)
    table = ""
    csv_path = os.path.join(HERE, "classification_report.csv")
    rep = None
    if os.path.exists(csv_path):
        rep = pd.read_csv(csv_path, index_col=0)
        rows = "".join(
            f"<tr><td>{html.escape(str(idx))}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
            f"<td>{r['f1-score']:.3f}</td><td>{int(r['support'])}</td></tr>"
            for idx, r in rep.iterrows() if idx != "accuracy")
        table = f"""<div class="sub-h">Per-stage results on the test set</div>
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Stage</th><th>Precision</th><th>Recall</th>
        <th>F1-score</th><th>Images</th></tr></thead><tbody>{rows}</tbody></table></div>"""
    n_test = ""
    if os.path.exists(csv_path):
        n_test = f"{int(rep.loc['macro avg', 'support']):,} "
    return f"""<div class="sub-h">Held-out test set: {n_test}images the model never saw during training</div>
               <div class="tiles">{tile_html}</div>{table}"""


def find_figure(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(HERE, "figures", p)))
        if hits:
            return hits[-1]
    return None


def how_it_works_html():
    steps = [
        ("1", "Crop and resize", f"The black border is cropped, the image is padded to a square and resized to "
                                 f"the input size of each model ({SIZES_LABEL}) with area interpolation."),
        ("2", "Preprocess", {"raw": "The image is passed on without enhancement: in the notebook's full-training "
                                     "comparison this scored best, because EfficientNet normalises the pixels itself.",
                             "clahe_unsharp": "Median denoising, CLAHE contrast enhancement and unsharp-mask edge "
                                              "enhancement make small lesions stand out.",
                             "ben_graham": "Local average colour is subtracted (Ben Graham's method) to even out "
                                           "lighting and highlight lesions."}.get(MODE, MODE)),
        ("3", "Classify", (f"An ImageNet-pretrained {BACKBONE}, fine-tuned on APTOS 2019, outputs a probability "
                           f"for each of the 5 stages" if len(MODELS) == 1 else
                           "Three ImageNet-pretrained CNNs fine-tuned on APTOS 2019 ("
                           + ", ".join(f"{html.escape(n)}" for n, _, _ in MODELS)
                           + ") each output a probability for every stage, and the three are averaged. The "
                             "combination was chosen on the validation set only")
                          + (" (averaged over 4 flipped views)." if USE_TTA else ".")),
        ("4", "Decide", "The most likely stage is reported. Any-DR probability = 1 &minus; P(No DR)."
                        if DECISION == "argmax" else "Stage thresholds tuned on the validation set turn the "
                                                      "expected grade into a stage."),
        ("5", "Explain", "Grad-CAM uses the gradients of the predicted stage to highlight the image regions "
                         "that most influenced the decision" + ("; the maps of all models are averaged."
                                                                if len(MODELS) > 1 else ".")),
    ]
    step_html = "".join(f'<div class="step"><span class="step-n">{n}</span><div><b>{t}</b><br>{d}</div></div>'
                        for n, t, d in steps)
    stage_rows = "".join(
        f'<tr><td><span class="dot" style="--c:{s[3]}"></span>{i}</td><td>{s[0]}</td><td>{s[1]}</td><td>{s[2]}</td></tr>'
        for i, s in enumerate(STAGES))
    return f"""<div class="steps">{step_html}</div>
      <div class="sub-h">The five stages (international clinical DR severity scale)</div>
      <div class="tbl-wrap"><table class="tbl"><thead><tr><th>#</th><th>Stage</th><th>What is seen</th>
      <th>Suggested next step</th></tr></thead><tbody>{stage_rows}</tbody></table></div>
      <div class="note"><b>Reading the heat map:</b> red and yellow areas influenced the prediction most, blue
      areas least. On DR images the heat should sit on haemorrhages, exudates or new vessels. Heat on the optic
      disc or image edges shows the model also uses normal anatomy, a known limitation.</div>"""


def about_html():
    return """<div class="about">
      <p><b>Project:</b> Diabetic Retinopathy Stage Detection using CNN transfer learning.<br>
      <b>Module:</b> Computer Vision, BSc (Hons) Computer Science, National Institute of Business Management.</p>
      <p><b>Data:</b> APTOS 2019 Blindness Detection (Kaggle): 3,662 retinal fundus photographs graded 0&ndash;4
      by clinicians, split 70 / 15 / 15 into training, validation and test sets.</p>
      <p><b>Limitations:</b> trained on one dataset from one region and one set of cameras; image quality and
      grader disagreement affect the labels; the Mild and Severe stages have few examples and are the hardest
      to recognise.</p>
      <div class="note warn"><b>Not a medical device.</b> This prototype was built for coursework. Its output
      must not be used to diagnose or treat anyone. Anyone with diabetes should have regular eye examinations
      by a qualified professional.</div></div>"""


# ---------------------------------------------------------------------------------------------
# 7. Styling
# ---------------------------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
footer {display: none !important;}
.hero {padding: 26px 30px; border-radius: 16px; margin-bottom: 6px;
       background: linear-gradient(120deg, #0b3b44 0%, #0f5561 55%, #16707c 100%); color: #eef7f7;}
.hero h1 {margin: 4px 0 6px; font-size: 30px; font-weight: 700; color: #fff; letter-spacing: -0.01em;}
.hero p {margin: 0 0 14px; max-width: 760px; color: #cfe6e8; font-size: 15px; line-height: 1.5;}
.hero-eyebrow {font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: #8fd3db; font-weight: 600;}
.chips {display: flex; flex-wrap: wrap; gap: 8px;}
.chip {background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); color: #f2fbfb;
       padding: 4px 11px; border-radius: 999px; font-size: 13px;}
.step-label {font-weight: 600; font-size: 14px; margin: 2px 0 -4px; color: var(--body-text-color-subdued);
             text-transform: uppercase; letter-spacing: .06em;}
#analyse-btn {font-size: 16px; min-height: 46px;}

.rc {border: 1px solid var(--border-color-primary); border-left: 6px solid var(--c, #16707c);
     border-radius: 14px; padding: 20px 22px; background: var(--block-background-fill);}
.rc-empty, .rc-error {border-left-width: 1px; text-align: center; padding: 48px 28px;}
.rc-error {border-color: #cf3b3b;}
.rc-empty-icon {font-size: 34px; color: #16707c; line-height: 1;}
.rc-empty-title {font-size: 18px; font-weight: 700; margin: 10px 0 6px;}
.rc-empty-text {color: var(--body-text-color-subdued); max-width: 460px; margin: auto; line-height: 1.55;}
.rc-head {display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}
.rc-pill {background: var(--c); color: #fff; font-weight: 700; font-size: 12px; letter-spacing: .06em;
          padding: 4px 10px; border-radius: 999px;}
.rc-verdict {font-weight: 600; font-size: 15px;}
.rc-stage {font-size: 28px; font-weight: 750; margin: 10px 0 2px; color: var(--c);}
.rc-desc {color: var(--body-text-color-subdued); margin-bottom: 14px;}
.rc-kpis {display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin-bottom: 16px;}
.rc-kpis > div {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 12px;}
.k-v {display: block; font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums;}
.k-l {display: block; font-size: 12.5px; color: var(--body-text-color-subdued);}
.rc-sec {font-size: 12px; text-transform: uppercase; letter-spacing: .07em; font-weight: 600;
         color: var(--body-text-color-subdued); margin: 6px 0 8px;}
.scale {display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 5px; margin-bottom: 16px;}
.seg {border-top: 6px solid var(--c); border-radius: 6px; padding: 7px 6px 6px; text-align: center;
      background: var(--background-fill-secondary); opacity: .55;}
.seg-on {opacity: 1; outline: 2px solid var(--c); outline-offset: 1px;}
.seg-n {display: block; font-weight: 700; font-size: 16px;}
.seg-t {display: block; font-size: 11.5px; line-height: 1.2; color: var(--body-text-color-subdued);}
.bars {display: grid; gap: 7px; margin-bottom: 16px;}
.bar-row {display: grid; grid-template-columns: 150px 1fr 58px; align-items: center; gap: 10px; font-size: 14px;}
.bar-on .bar-label, .bar-on .bar-pct {font-weight: 700;}
.bar-track {height: 10px; border-radius: 6px; background: var(--background-fill-secondary); overflow: hidden;}
.bar-fill {display: block; height: 100%; border-radius: 6px; background: var(--c);}
.bar-pct {text-align: right; font-variant-numeric: tabular-nums;}
.rc-action {border-radius: 10px; padding: 12px 14px; background: var(--background-fill-secondary); line-height: 1.5;}
.rc-foot {margin-top: 12px; font-size: 12px; color: var(--body-text-color-subdued); line-height: 1.5;}

.tiles {display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 10px 0 18px;}
.tile {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 14px 16px;
       background: var(--block-background-fill);}
.t-l {font-size: 12.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--body-text-color-subdued); font-weight: 600;}
.t-v {font-size: 28px; font-weight: 750; margin: 2px 0; font-variant-numeric: tabular-nums; color: #16707c;}
.t-s {font-size: 13px; color: var(--body-text-color-subdued);}
.sub-h {font-weight: 700; font-size: 16px; margin: 14px 0 8px;}
.tbl-wrap {overflow-x: auto;}
.tbl {width: 100%; border-collapse: collapse; font-size: 14px;}
.tbl th {text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: .05em;
         color: var(--body-text-color-subdued); border-bottom: 1px solid var(--border-color-primary); padding: 8px 10px;}
.tbl td {border-bottom: 1px solid var(--border-color-primary); padding: 8px 10px; vertical-align: top;}
.tbl td:first-child {white-space: nowrap;}
#examples .grid-container {grid-template-columns: repeat(5, minmax(0, 1fr)) !important;}
#examples .caption-label, #examples figcaption {font-size: 12px;}
.members {margin-bottom: 16px;}
.members td, .members th {padding: 6px 10px;}
.dot {display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--c); margin-right: 8px;}
.steps {display: grid; gap: 10px; margin-bottom: 8px;}
.step {display: flex; gap: 12px; align-items: flex-start; border: 1px solid var(--border-color-primary);
       border-radius: 12px; padding: 12px 14px; line-height: 1.5;}
.step-n {flex: none; width: 28px; height: 28px; border-radius: 50%; background: #16707c; color: #fff;
         display: grid; place-items: center; font-weight: 700;}
.note {margin-top: 14px; border-radius: 10px; padding: 12px 14px; background: var(--background-fill-secondary); line-height: 1.55;}
.note.warn {border-left: 4px solid #d99a14;}
.about p {line-height: 1.6; max-width: 900px;}
@media (max-width: 720px) {
  .rc-kpis, .tiles {grid-template-columns: 1fr;}
  .bar-row {grid-template-columns: 110px 1fr 52px;}
  .seg-t {display: none;}
}
"""

THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("IBM Plex Sans"), "Segoe UI", "system-ui", "sans-serif"])

# ---------------------------------------------------------------------------------------------
# 8. Layout
# ---------------------------------------------------------------------------------------------
examples_dir = os.path.join(HERE, "examples")
example_files = sorted(glob.glob(os.path.join(examples_dir, "*.png")) + glob.glob(os.path.join(examples_dir, "*.jp*g")))


def example_label(path):
    name = os.path.basename(path)
    if name.startswith("stage") and name[5:6].isdigit():
        s = int(name[5])
        return f"Stage {s}"
    return name


with gr.Blocks(title="DR Stage Detection") as demo:
    gr.HTML(header_html())
    with gr.Tabs():
        # ---------------- Tab 1: analyse an image ----------------
        with gr.Tab("Analyse an image"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    gr.HTML('<div class="step-label">1 &middot; Choose a fundus photograph</div>')
                    inp = gr.Image(type="numpy", label="Fundus image", height=340,
                                   sources=["upload", "clipboard"])
                    with gr.Row():
                        btn = gr.Button("Analyse image", variant="primary", elem_id="analyse-btn", scale=3)
                        clr = gr.Button("Clear", variant="secondary", scale=1)
                    if example_files:
                        gr.HTML('<div class="step-label">Or click an unseen test image (true stage shown)</div>')
                        gallery = gr.Gallery(value=[(p, example_label(p)) for p in example_files],
                                             columns=5, rows=2, height=250, allow_preview=False, show_label=False,
                                             object_fit="cover", elem_id="examples")
                with gr.Column(scale=7):
                    gr.HTML('<div class="step-label">2 &middot; Result</div>')
                    result = gr.HTML(placeholder_card())
            gr.HTML('<div class="step-label">3 &middot; What the model saw and where it looked</div>')
            with gr.Row():
                orig_out = gr.Image(label="Original (cropped)", height=300, interactive=False)
                pre_out = gr.Image(label="Model input", height=300, interactive=False)
                cam_out = gr.Image(label="Grad-CAM heat map", height=300, interactive=False)

        # ---------------- Tab 2: model performance ----------------
        with gr.Tab("Model performance"):
            gr.HTML(metrics_html())
            figs = [("Ensemble selection on the validation set", find_figure("*ensemble_selection*.png")),
                    ("Confusion matrix", find_figure("*confusion_matrix*.png")),
                    ("Accuracy, loss and QWK during training", find_figure("*curves*.png")),
                    ("ROC curves and DR vs No DR confusion matrix", find_figure("*roc_binary*.png")),
                    ("Precision, recall and F1 per stage", find_figure("*per_class_prf*.png"))]
            figs = [(t, p) for t, p in figs if p]
            for title, path in figs:
                gr.HTML(f'<div class="sub-h">{title}</div>')
                gr.Image(value=path, show_label=False, interactive=False, container=False)

        # ---------------- Tab 3: how it works ----------------
        with gr.Tab("How it works"):
            gr.HTML(how_it_works_html())
            gc_fig = find_figure("*gradcam*.png")
            if gc_fig:
                gr.HTML('<div class="sub-h">Grad-CAM on one test image per stage (from the notebook)</div>')
                gr.Image(value=gc_fig, show_label=False, interactive=False, container=False)

        # ---------------- Tab 4: about ----------------
        with gr.Tab("About"):
            gr.HTML(about_html())

    btn.click(analyse, inputs=inp, outputs=[result, orig_out, pre_out, cam_out])
    if example_files:
        def pick(evt: gr.SelectData):
            return cv2.cvtColor(cv2.imread(example_files[evt.index]), cv2.COLOR_BGR2RGB)
        gallery.select(pick, outputs=inp).then(analyse, inputs=inp, outputs=[result, orig_out, pre_out, cam_out])
    inp.upload(lambda: (placeholder_card(), None, None, None), outputs=[result, orig_out, pre_out, cam_out])
    clr.click(clear, outputs=[inp, result, orig_out, pre_out, cam_out])


if __name__ == "__main__":
    # theme/css are launch() options in Gradio 6 and Blocks() options in older versions
    launch_params = inspect.signature(demo.launch).parameters
    extra = {k: v for k, v in {"theme": THEME, "css": CSS}.items() if k in launch_params}
    if len(extra) < 2:
        demo.theme, demo.css = THEME, CSS
    on_server = bool(os.environ.get("SPACE_ID"))        # set automatically on Hugging Face Spaces
    demo.launch(inbrowser=not on_server, **extra)
