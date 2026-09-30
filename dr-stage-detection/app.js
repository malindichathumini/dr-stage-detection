/*
 * Diabetic Retinopathy Stage Detection - in-browser inference (Version 4: ensemble of 3 CNNs)
 * -----------------------------------------------------------------------------------------
 * Every model in the ensemble is split in two:
 *   - its EfficientNet backbone runs as an ONNX model in ONNX Runtime Web (WebAssembly);
 *     large backbones are stored as several parts under 25 MB and joined here after download;
 *   - its small classification head (GAP -> Dense 256 ReLU -> Dense 5 softmax) and Grad-CAM are
 *     computed here in JavaScript from the exported weights (head.bin).
 * Each model gets the image at its own input size (300 or 380 px), the stage probabilities of the
 * models are averaged, and the Grad-CAM maps are averaged. Nothing is uploaded: the image stays on
 * the device.
 */
"use strict";

const STAGES = [
  ["No DR", "No visible signs of diabetic retinopathy.",
   "Routine screening again in 12 months. Keep blood sugar, blood pressure and cholesterol under control.", "#2f9e6f"],
  ["Mild NPDR", "Microaneurysms only: tiny bulges in the retinal blood vessels.",
   "Re-screen in 6 to 12 months and review diabetes control with the patient's doctor.", "#8fae2b"],
  ["Moderate NPDR", "More microaneurysms, dot/blot haemorrhages and hard exudates.",
   "Refer to an ophthalmologist within about 3 months.", "#d99a14"],
  ["Severe NPDR", "Many haemorrhages in all quadrants, venous beading or abnormal vessels (4-2-1 rule).",
   "Urgent referral to an ophthalmologist within weeks: high risk of progressing to proliferative DR.", "#e0662a"],
  ["Proliferative DR", "New fragile blood vessels grow on the retina and can bleed or detach it.",
   "Urgent referral: treatment such as laser photocoagulation or anti-VEGF injections may be needed.", "#cf3b3b"],
];

const $ = (id) => document.getElementById(id);
let CFG = null, MEMBERS = [], READY = false, currentImage = null;

// Older single-model config (Version 2) -> list with one member
function memberList(cfg) {
  if (cfg.members) return cfg.members;
  return [{ name: cfg.backbone, img_size: cfg.img_size, onnx: ["model/backbone.onnx"], head: "model/head.bin",
            head_dims: cfg.head, feature_shape: cfg.feature_shape }];
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function fetchJoined(urls) {
  // download all parts of one ONNX file and join them into one buffer
  const parts = await Promise.all(urls.map(fetchBytes));
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// ------------------------------------------------------------------ loading
async function init() {
  try {
    CFG = await (await fetch("model/config.json", { cache: "no-cache" })).json();   // always the newest settings
    renderFacts(); renderMetrics(); showPlaceholder();

    ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
    // several threads only when the page is cross-origin isolated (see vercel.json), otherwise one
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

    const list = memberList(CFG);
    for (let i = 0; i < list.length; i++) {
      const m = list[i];
      setStatus("", `Loading model ${i + 1} of ${list.length}: ${m.name} (${m.size_mb || "?"} MB, only the first time)…`);
      const all = new Float32Array((await fetchBytes(m.head)).buffer);
      const { in: nIn, hidden: nH, out: nOut } = m.head_dims;
      let o = 0;
      const take = (n) => { const a = all.subarray(o, o + n); o += n; return a; };
      const head = { W1: take(nIn * nH), b1: take(nH), W2: take(nH * nOut), b2: take(nOut), nIn, nH, nOut };
      const session = await ort.InferenceSession.create(await fetchJoined(m.onnx), { executionProviders: ["wasm"] });
      MEMBERS.push({ ...m, head, session, inputName: session.inputNames[0] });
    }
    READY = true;
    setStatus("ok", `${MEMBERS.length > 1 ? "All " + MEMBERS.length + " models" : "Model"} ready. Runs on your device; the image is never uploaded.`);
    $("analyse").disabled = !currentImage;
  } catch (e) {
    console.error(e);
    setStatus("err", "The model could not be loaded: " + e.message);
  }
}

function setStatus(kind, text) {
  $("status").className = "status " + kind;
  $("status-text").textContent = text;
  // short version in the sidebar, visible on every page
  $("model-state").className = "model-state " + kind;
  $("model-state-text").textContent = kind === "ok" ? `${MEMBERS.length} models ready` :
    kind === "err" ? "Models failed to load" : text.replace(/ \(.*$/, "").replace(/…$/, "");
}

// ------------------------------------------------------------------ image input
function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    $("preview").src = url; $("preview").hidden = false; $("drop-empty").hidden = true;
    $("analyse").disabled = !READY;
    showPlaceholder(); clearCanvases(); $("lens-wrap").hidden = true;
  };
  img.src = url;
}

$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));
$("clear").addEventListener("click", () => {
  currentImage = null; $("file").value = ""; $("preview").hidden = true; $("drop-empty").hidden = false;
  $("analyse").disabled = true; showPlaceholder(); clearCanvases(); $("lens-wrap").hidden = true;
});
$("analyse").addEventListener("click", analyse);

// ------------------------------------------------------------------ preprocessing (same as the notebook)
function readPixels(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return { data: ctx.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
}

function cropBox(px, tol = 7) {
  // rows / columns that contain a pixel brighter than `tol` (OpenCV grey = 0.299R + 0.587G + 0.114B)
  const { data, w, h } = px;
  let top = h, bottom = -1, left = w, right = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const g = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      if (g > tol) {
        if (y < top) top = y; if (y > bottom) bottom = y;
        if (x < left) left = x; if (x > right) right = x;
      }
    }
  }
  if (bottom < 0) return { x0: 0, y0: 0, w, h };
  return { x0: left, y0: top, w: right - left + 1, h: bottom - top + 1 };
}

function areaWeights(srcLen, dstLen) {
  // INTER_AREA: each output pixel averages the source interval it covers (fractional overlaps weighted)
  const scale = srcLen / dstLen, out = [];
  for (let d = 0; d < dstLen; d++) {
    const a = d * scale, b = (d + 1) * scale, idx = [], wts = [];
    for (let s = Math.floor(a); s < Math.min(Math.ceil(b), srcLen); s++) {
      const ov = Math.min(b, s + 1) - Math.max(a, s);
      if (ov > 1e-9) { idx.push(s); wts.push(ov / scale); }
    }
    out.push({ idx, wts });
  }
  return out;
}

function preprocess(px, S) {
  // crop black border -> pad to square (black) -> resize to S x S with area interpolation -> uint8 RGB
  const box = cropBox(px);
  const side = Math.max(box.w, box.h);
  const offY = Math.floor((side - box.h) / 2), offX = Math.floor((side - box.w) / 2);
  const get = (sy, sx, ch) => {             // pixel of the padded square image
    const y = sy - offY, x = sx - offX;
    if (y < 0 || x < 0 || y >= box.h || x >= box.w) return 0;
    return px.data[((box.y0 + y) * px.w + (box.x0 + x)) * 4 + ch];
  };
  const out = new Uint8ClampedArray(S * S * 3);
  if (side >= S) {
    const W = areaWeights(side, S);
    const tmp = new Float32Array(side * S * 3);          // horizontal pass
    for (let y = 0; y < side; y++) {
      for (let dx = 0; dx < S; dx++) {
        const { idx, wts } = W[dx];
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < idx.length; k++) {
          r += get(y, idx[k], 0) * wts[k]; g += get(y, idx[k], 1) * wts[k]; b += get(y, idx[k], 2) * wts[k];
        }
        const o = (y * S + dx) * 3; tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
      }
    }
    for (let dy = 0; dy < S; dy++) {                     // vertical pass
      const { idx, wts } = W[dy];
      for (let x = 0; x < S; x++) {
        let r = 0, g = 0, b = 0;
        for (let k = 0; k < idx.length; k++) {
          const o = (idx[k] * S + x) * 3;
          r += tmp[o] * wts[k]; g += tmp[o + 1] * wts[k]; b += tmp[o + 2] * wts[k];
        }
        const o = (dy * S + x) * 3; out[o] = Math.round(r); out[o + 1] = Math.round(g); out[o + 2] = Math.round(b);
      }
    }
  } else {                                               // small images: bilinear upscaling
    const c = document.createElement("canvas"); c.width = c.height = S;
    const src = document.createElement("canvas"); src.width = src.height = side;
    const sctx = src.getContext("2d"); sctx.fillStyle = "#000"; sctx.fillRect(0, 0, side, side);
    const id = sctx.createImageData(box.w, box.h);
    for (let y = 0; y < box.h; y++) for (let x = 0; x < box.w; x++) for (let ch = 0; ch < 4; ch++)
      id.data[(y * box.w + x) * 4 + ch] = ch === 3 ? 255 : px.data[((box.y0 + y) * px.w + box.x0 + x) * 4 + ch];
    sctx.putImageData(id, offX, offY);
    const ctx = c.getContext("2d"); ctx.drawImage(src, 0, 0, S, S);
    const d = ctx.getImageData(0, 0, S, S).data;
    for (let i = 0; i < S * S; i++) { out[i * 3] = d[i * 4]; out[i * 3 + 1] = d[i * 4 + 1]; out[i * 3 + 2] = d[i * 4 + 2]; }
  }
  return { rgb: out, box, side, offX, offY };
}

// ------------------------------------------------------------------ model: backbone (ONNX) + head (JS)
function flipped(rgb, S, lr, ud) {
  const out = new Float32Array(S * S * 3);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const sy = ud ? S - 1 - y : y, sx = lr ? S - 1 - x : x;
    const s = (sy * S + sx) * 3, d = (y * S + x) * 3;
    out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2];
  }
  return out;
}

async function features(member, rgbFloat, S) {
  const t = new ort.Tensor("float32", rgbFloat, [1, S, S, 3]);
  const res = await member.session.run({ [member.inputName]: t });
  return res[member.session.outputNames[0]].data;   // [1, h, w, channels] NHWC, e.g. 10x10x1280 for B0
}

function head(HEAD, feat) {
  const { W1, b1, W2, b2, nIn, nH, nOut } = HEAD;
  const P = feat.length / nIn;
  const g = new Float32Array(nIn);
  for (let p = 0; p < P; p++) for (let k = 0; k < nIn; k++) g[k] += feat[p * nIn + k];
  for (let k = 0; k < nIn; k++) g[k] /= P;             // global average pooling
  const h = new Float32Array(nH);
  for (let i = 0; i < nH; i++) h[i] = b1[i];
  for (let k = 0; k < nIn; k++) { const gk = g[k]; if (gk === 0) continue; const row = k * nH; for (let i = 0; i < nH; i++) h[i] += gk * W1[row + i]; }
  for (let i = 0; i < nH; i++) h[i] = Math.max(0, h[i]);   // ReLU
  const z = new Float64Array(nOut);
  for (let j = 0; j < nOut; j++) { let s = b2[j]; for (let i = 0; i < nH; i++) s += h[i] * W2[i * nOut + j]; z[j] = s; }
  const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const sum = e.reduce((a, b) => a + b, 0);
  return { probs: Array.from(e, (v) => v / sum), h };
}

function gradCam(member, feat, probs, h, cls) {
  // Grad-CAM with gradients derived analytically through the dense head (identical to the TF version)
  const { W1, W2, nIn, nH, nOut } = member.head;
  const dz = probs.map((p, j) => probs[cls] * ((j === cls ? 1 : 0) - p));   // d softmax_c / d logits
  const dh = new Float64Array(nH);
  for (let i = 0; i < nH; i++) { if (h[i] <= 0) continue; let s = 0; for (let j = 0; j < nOut; j++) s += W2[i * nOut + j] * dz[j]; dh[i] = s; }
  const w = new Float64Array(nIn);                     // channel weights
  for (let k = 0; k < nIn; k++) { let s = 0; const row = k * nH; for (let i = 0; i < nH; i++) s += W1[row + i] * dh[i]; w[k] = s; }
  const [fh, fw] = member.feature_shape;
  const cam = new Float64Array(fh * fw); let mx = 0;
  for (let p = 0; p < fh * fw; p++) { let s = 0; for (let k = 0; k < nIn; k++) s += feat[p * nIn + k] * w[k]; cam[p] = Math.max(0, s); mx = Math.max(mx, cam[p]); }
  for (let p = 0; p < cam.length; p++) cam[p] /= (mx + 1e-8);
  return { cam, fh, fw };
}

function decide(probs) {
  if (CFG.decision === "thresholds") {
    const grade = probs.reduce((a, p, i) => a + p * i, 0);
    return CFG.thresholds.filter((t) => grade >= t).length;
  }
  return probs.indexOf(Math.max(...probs));
}

// ------------------------------------------------------------------ drawing
function jet(v) {
  const c = (x) => Math.max(0, Math.min(1, x));
  return [c(1.5 - Math.abs(4 * v - 3)) * 255, c(1.5 - Math.abs(4 * v - 2)) * 255, c(1.5 - Math.abs(4 * v - 1)) * 255];
}

function drawRGB(canvas, rgb, S) {
  canvas.width = canvas.height = S;
  const ctx = canvas.getContext("2d"), id = ctx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) { id.data[i * 4] = rgb[i * 3]; id.data[i * 4 + 1] = rgb[i * 3 + 1]; id.data[i * 4 + 2] = rgb[i * 3 + 2]; id.data[i * 4 + 3] = 255; }
  ctx.putImageData(id, 0, 0);
}

function upsample({ cam, fh, fw }, S) {
  // bilinear upsampling of a small CAM (e.g. 10x10) to S x S (pixel-centre aligned, like cv2.resize)
  const out = new Float32Array(S * S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const fy = Math.min(Math.max((y + 0.5) * fh / S - 0.5, 0), fh - 1), fx = Math.min(Math.max((x + 0.5) * fw / S - 0.5, 0), fw - 1);
    const y0 = Math.floor(fy), x0 = Math.floor(fx), y1 = Math.min(y0 + 1, fh - 1), x1 = Math.min(x0 + 1, fw - 1);
    const ay = fy - y0, ax = fx - x0;
    out[y * S + x] = (1 - ay) * ((1 - ax) * cam[y0 * fw + x0] + ax * cam[y0 * fw + x1]) + ay * ((1 - ax) * cam[y1 * fw + x0] + ax * cam[y1 * fw + x1]);
  }
  return out;
}

function drawCam(canvas, rgb, S, full) {
  // full: S x S heat values in [0, 1]
  canvas.width = canvas.height = S;
  const ctx = canvas.getContext("2d"), id = ctx.createImageData(S, S);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const i = y * S + x, [r, g, b] = jet(full[i]);
    id.data[i * 4] = 0.55 * rgb[i * 3] + 0.45 * r;
    id.data[i * 4 + 1] = 0.55 * rgb[i * 3 + 1] + 0.45 * g;
    id.data[i * 4 + 2] = 0.55 * rgb[i * 3 + 2] + 0.45 * b;
    id.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

function drawOriginal(canvas, img, pre) {
  const S = 600; canvas.width = canvas.height = S;
  const ctx = canvas.getContext("2d"); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, S, S);
  const k = S / pre.side;
  ctx.drawImage(img, pre.box.x0, pre.box.y0, pre.box.w, pre.box.h, pre.offX * k, pre.offY * k, pre.box.w * k, pre.box.h * k);
}

function clearCanvases() {
  ["c-orig", "c-input", "c-cam"].forEach((id) => { const c = $(id); c.getContext("2d").clearRect(0, 0, c.width, c.height); });
}

// ------------------------------------------------------------------ main action
async function analyse() {
  if (!currentImage || !READY) return;
  $("analyse").disabled = true; $("analyse").textContent = "Analysing…";
  await new Promise((r) => setTimeout(r, 20));        // let the button repaint
  try {
    const t0 = performance.now();
    const px = readPixels(currentImage);
    const pres = {};                                   // one preprocessed image per input size
    for (const m of MEMBERS) if (!pres[m.img_size]) pres[m.img_size] = preprocess(px, m.img_size);
    const views = tta() ? [[0, 0], [1, 0], [0, 1], [1, 1]] : [[0, 0]];

    // 1. every model predicts; its probabilities are averaged over the views
    const runs = [];
    for (let i = 0; i < MEMBERS.length; i++) {
      const m = MEMBERS[i], S = m.img_size, pre = pres[S];
      $("analyse").textContent = `Analysing… model ${i + 1} of ${MEMBERS.length}`;
      await new Promise((r) => setTimeout(r, 0));
      let p = null, feat0 = null, h0 = null, p0 = null;
      for (const [lr, ud] of views) {
        const f = await features(m, flipped(pre.rgb, S, lr, ud), S);
        const r = head(m.head, f);
        if (!feat0) { feat0 = f; h0 = r.h; p0 = r.probs; }
        p = p ? p.map((v, k) => v + r.probs[k]) : r.probs;
      }
      runs.push({ m, probs: p.map((v) => v / views.length), feat0, h0, p0 });
    }

    // 2. ensemble = mean of the models' probabilities
    const probs = runs[0].probs.map((_, k) => runs.reduce((a, r) => a + r.probs[k], 0) / runs.length);
    const stage = decide(probs);
    const cls = probs.indexOf(Math.max(...probs));

    // 3. Grad-CAM of every model for the ensemble's stage, shown on the largest input and averaged
    const D = Math.max(...Object.keys(pres).map(Number)), show = pres[D];
    const heat = new Float32Array(D * D);
    for (const r of runs) {
      const up = upsample(gradCam(r.m, r.feat0, r.p0, r.h0, cls), D);
      for (let i = 0; i < heat.length; i++) heat[i] += up[i];
    }
    let mx = 0; for (const v of heat) mx = Math.max(mx, v);
    for (let i = 0; i < heat.length; i++) heat[i] /= (mx + 1e-8);

    drawOriginal($("c-orig"), currentImage, show);
    drawRGB($("c-input"), show.rgb, D);
    drawCam($("c-cam"), show.rgb, D, heat);
    $("lens-wrap").hidden = false;
    $("result").innerHTML = resultCard(probs, stage, (performance.now() - t0) / 1000, runs);
    document.body.dataset.probs = JSON.stringify(probs);        // used by automated tests
    document.body.dataset.memberProbs = JSON.stringify(runs.map((r) => r.probs));
  } catch (e) {
    console.error(e);
    $("result").innerHTML = `<div class="rc rc-empty"><div class="rc-empty-title">Could not analyse this image</div>
      <div class="rc-empty-text">${escapeHtml(e.message)}</div></div>`;
  } finally {
    $("analyse").disabled = false; $("analyse").textContent = "Analyse image";
  }
}

// ------------------------------------------------------------------ HTML pieces
const pct = (v) => (v * 100).toFixed(1) + "%";
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function showPlaceholder() {
  $("result").innerHTML = `<div class="rc rc-empty"><div class="rc-empty-icon">&#9673;</div>
    <div class="rc-empty-title">No image analysed yet</div>
    <div class="rc-empty-text">Choose a retinal fundus photograph, then press <b>Analyse image</b>. The stage, the
    probability of each stage and a heat map of where the model looked will appear here.</div></div>`;
}

const tta = () => (CFG.ensemble_tta !== undefined ? CFG.ensemble_tta : CFG.tta);
const modelLabel = () => (MEMBERS.length > 1 || (CFG.members || []).length > 1 ? `Ensemble of ${memberList(CFG).length} CNNs` : CFG.backbone);
const sizeLabel = () => [...new Set(memberList(CFG).map((m) => m.img_size))].sort().map((s) => s + "px").join(" / ");

function membersTable(runs, stage) {
  // how each model in the ensemble voted
  if (!runs || runs.length < 2) return "";
  const rows = runs.map((r) => {
    const s = r.probs.indexOf(Math.max(...r.probs));
    return `<tr><td>${escapeHtml(r.m.name)}</td><td>${r.m.img_size}px</td><td>${s} &middot; ${STAGES[s][0]}</td><td>${pct(r.probs[stage])}</td></tr>`;
  }).join("");
  return `<div class="rc-sec">How each model in the ensemble voted</div>
    <div class="tbl-wrap"><table class="tbl members"><thead><tr><th>Model</th><th>Input</th><th>Its stage</th><th>P(stage ${stage})</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

function resultCard(probs, stage, seconds, runs) {
  const [name, desc, action, colour] = STAGES[stage];
  const pDr = 1 - probs[0], dr = pDr >= 0.5;
  const scale = STAGES.map((s, i) => `<div class="seg${i === stage ? " on" : ""}" style="--c:${s[3]}">
      <span class="seg-n">${i}</span><span class="seg-t">${s[0]}</span></div>`).join("");
  const bars = STAGES.map((s, i) => `<div class="bar-row${i === stage ? " on" : ""}">
      <span>${i} &middot; ${s[0]}</span><span class="bar-track"><span class="bar-fill" style="width:${(probs[i] * 100).toFixed(1)}%;--c:${s[3]}"></span></span>
      <span class="bar-pct">${pct(probs[i])}</span></div>`).join("");
  return `<div class="rc" style="--c:${colour}">
    <div class="rc-head"><span class="rc-pill">${dr ? "DR" : "NO DR"}</span>
      <span class="rc-verdict">${dr ? "Diabetic retinopathy detected" : "No diabetic retinopathy detected"}</span></div>
    <div class="rc-stage">Stage ${stage} &middot; ${name}</div>
    <div class="rc-desc">${desc}</div>
    <div class="kpis">
      <div><span class="k-v">${pct(probs[stage])}</span><span class="k-l">confidence in this stage</span></div>
      <div><span class="k-v">${pct(pDr)}</span><span class="k-l">probability of any DR</span></div>
      <div><span class="k-v">${seconds.toFixed(1)} s</span><span class="k-l">analysis time on this device</span></div>
    </div>
    <div class="rc-sec">Severity scale</div><div class="scale">${scale}</div>
    <div class="rc-sec">Probability of each stage</div><div class="bars">${bars}</div>
    ${membersTable(runs, stage)}
    <div class="rc-action"><b>Suggested next step:</b> ${action}</div>
    <div class="rc-foot">Research prototype for coursework. It is not a medical device and must not be used for
      diagnosis. Model: ${modelLabel()} &middot; input ${sizeLabel()} &middot;
      preprocessing: ${CFG.preprocess_mode} &middot; TTA: ${tta() ? "on" : "off"} &middot; decision: ${CFG.decision}</div>
  </div>`;
}

function renderFacts() {
  // headline numbers on the overview page, read from model/config.json
  const m = CFG.test_stage_metrics, b = CFG.test_binary_metrics;
  const facts = [[pct(m.accuracy), "stage accuracy on 550 unseen test images"],
    [m.qwk.toFixed(3), "quadratic weighted kappa, the official APTOS metric"],
    [pct(b["sensitivity (recall)"]), "of eyes with retinopathy correctly flagged"],
    [String(memberList(CFG).length), "CNNs in the ensemble, running on this device"]];
  $("facts").innerHTML = facts.map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join("");
}

function renderMetrics() {
  const m = CFG.test_stage_metrics, b = CFG.test_binary_metrics;
  const tiles = [
    ["Stage accuracy", pct(m.accuracy), "exact stage correct"],
    ["Macro F1", m.macro_f1.toFixed(3), "average over the 5 stages"],
    ["QWK", m.qwk.toFixed(3), "agreement with graders (official APTOS metric)"],
    ["Sensitivity", pct(b["sensitivity (recall)"]), "DR cases correctly flagged"],
    ["Specificity", pct(b.specificity), "healthy eyes correctly cleared"],
    ["ROC AUC", b.roc_auc.toFixed(3), "DR vs No DR separation"],
  ].map(([t, v, s]) => `<div class="tile"><div class="t-l">${t}</div><div class="t-v">${v}</div><div class="t-s">${s}</div></div>`).join("");
  const rows = CFG.per_class.map((r) => `<tr><td><span class="dot" style="--c:${STAGES[r.stage][3]}"></span>${r.stage} &middot; ${STAGES[r.stage][0]}</td>
      <td>${r.precision.toFixed(3)}</td><td>${r.recall.toFixed(3)}</td><td>${r.f1.toFixed(3)}</td><td>${r.support}</td></tr>`).join("");
  $("metrics").innerHTML = `<div class="tiles">${tiles}</div>
    <div class="sub-h">Per-stage results on the test set</div>
    <div class="tbl-wrap"><table class="tbl"><thead><tr><th>Stage</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Images</th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

// ------------------------------------------------------------------ pages (hash navigation)
const VIEWS = [...document.querySelectorAll(".view")].map((v) => v.id.replace("view-", ""));
function route() {
  const name = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== "view-" + name; });
  document.querySelectorAll(".nav a").forEach((a) => {
    if (a.dataset.view === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  document.title = (name === "overview" ? "" : document.getElementById("view-" + name).dataset.title + " | ") + "DR Stage Detection";
  closeMenu();
  window.scrollTo(0, 0);
  $("main").focus({ preventScroll: true });
}
window.addEventListener("hashchange", route);

// mobile menu
function closeMenu() { document.body.classList.remove("nav-open"); $("scrim").hidden = true; $("menu-btn").setAttribute("aria-expanded", "false"); }
$("menu-btn").addEventListener("click", () => {
  const open = !document.body.classList.contains("nav-open");
  document.body.classList.toggle("nav-open", open); $("scrim").hidden = !open; $("menu-btn").setAttribute("aria-expanded", String(open));
});
$("scrim").addEventListener("click", closeMenu);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });

// light / dark theme (remembered on this browser)
function currentTheme() {
  return document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function paintThemeButton() { $("theme-label").textContent = currentTheme() === "dark" ? "Light theme" : "Dark theme"; }
$("theme-btn").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked: theme lasts for this visit */ }
  paintThemeButton();
});
paintThemeButton();

// lens: drag to compare the photo with the heat map
const lens = $("lens"), slider = $("lens-slider");
function setCut(p) { p = Math.max(0, Math.min(100, p)); slider.value = p; lens.style.setProperty("--cut", p + "%"); }
slider.addEventListener("input", () => setCut(+slider.value));
let dragging = false;
const fromPointer = (e) => { const r = lens.getBoundingClientRect(); setCut(((e.clientX - r.left) / r.width) * 100); };
lens.addEventListener("pointerdown", (e) => { dragging = true; lens.setPointerCapture(e.pointerId); fromPointer(e); });
lens.addEventListener("pointermove", (e) => { if (dragging) fromPointer(e); });
lens.addEventListener("pointerup", () => { dragging = false; });

route();
init();
