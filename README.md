# Diabetic Retinopathy Stage Detection

A convolutional neural network that grades retinal fundus photographs on the international 5-stage
diabetic retinopathy (DR) scale, with Grad-CAM explanations and a web prototype that runs entirely in the
browser.

## Links

| What | Link |
|---|---|
| Live website (runs in your browser) | https://dr-stage-detection.vercel.app |
| Code (this repository) | https://github.com/malindichathumini/dr-stage-detection |
| Final results notebook: test evaluation (Kaggle) | https://www.kaggle.com/code/malindichathumini/dr-stage-detection-final-results |
| Experiments notebook: all 4 experiments + training (Kaggle) | https://www.kaggle.com/code/malindichathumini/dr-stage-detection-experiments |
| Video demonstration | https://YOUR-VIDEO-LINK |

NIBM · BSc (Hons) Computer Science · Computer Vision · Coursework 1

## Results (held-out test set, 550 images)

| Metric | Value |
|---|---|
| Stage accuracy | 78.9% |
| Macro F1 | 0.632 |
| Quadratic weighted kappa (QWK) | 0.879 |
| DR vs No DR sensitivity / specificity | 97.5% / 97.4% |
| DR vs No DR ROC AUC | 0.989 |

Version 1 of the pipeline (224 px, top-30% fine-tuning, loss-based model selection) reached 74.4% accuracy,
0.564 macro F1 and 0.826 QWK. Version 2 improved all three.

## Repository layout

| Path | What it is |
|---|---|
| `index.html`, `style.css`, `app.js` | The web prototype (static site, deployed on Vercel) |
| `model/backbone.onnx` | EfficientNetB0 feature extractor converted to ONNX (16 MB) |
| `model/head.bin` | Weights of the classification head (Dense 256 + Dense 5) |
| `model/config.json` | Image size, class names, decision rule and test metrics |
| `vendor/` | ONNX Runtime Web 1.22 (WebAssembly), self-hosted |
| `assets/figures/` | Evaluation figures shown on the site |
| `notebook/DR_Stage_Detection.ipynb` | Full training and evaluation pipeline (Kaggle, GPU) |
| `local_app/` | Gradio desktop version of the prototype |
| `tools/convert_to_onnx.py` | Converts the trained Keras model into the web files |

## Pipeline

1. **Data:** APTOS 2019 Blindness Detection (Kaggle), 3,662 images graded 0–4; stratified 70 / 15 / 15 split made before augmentation.
2. **Preprocessing:** black-border crop, pad to square, 300×300 area resize. CLAHE + unsharp masking and Ben Graham filtering were implemented and compared with full training; raw images scored best (validation QWK 0.861 vs 0.858 and 0.829).
3. **Augmentation:** flips, 360° rotation, zoom, translation, brightness and contrast (training only).
4. **Class balancing:** balanced weights, square-root weights and oversampling compared; oversampling selected.
5. **Model:** ImageNet EfficientNetB0 (selected against EfficientNetB3, ResNet50V2, MobileNetV2) + GAP → Dropout → Dense 256 (L2) → Dropout → Softmax 5.
6. **Training:** phase 1 trains the head with the backbone frozen; phase 2 fine-tunes the whole backbone (BatchNorm frozen) at 2e-5. Early stopping and checkpointing on validation QWK; learning rate reduced on plateau; label smoothing 0.05.
7. **Inference:** TTA and QWK-optimised thresholds were evaluated; plain argmax was kept because it gave the best balance of accuracy, macro F1 and QWK on the validation set (thresholds cut Mild recall to 12.5%).
8. **Explainability:** Grad-CAM on the last convolutional feature map.

## How the web version works

The Keras model is split in two. The EfficientNetB0 backbone is exported to ONNX and runs in the browser
with ONNX Runtime Web (WebAssembly). The small classification head and Grad-CAM are computed in
`app.js` from the exported weights, with Grad-CAM gradients derived analytically through the dense layers.
The browser output matches the Keras model to four decimal places. Because inference happens on the user's
device, the retinal image is never uploaded, which protects patient privacy and needs no paid server.

## Run it

**Website locally:** any static server, for example `python -m http.server 8000`, then open http://localhost:8000.

**Desktop app:** see `local_app/`. Put `dr_model.keras`, `dr_config.json`, `results.json` and
`classification_report.csv` from the notebook output next to `app.py`, install `requirements.txt`, then run
`python app.py` (or `start_app.bat` on Windows).

**Retrain:** import `notebook/DR_Stage_Detection.ipynb` into Kaggle, attach the APTOS 2019 competition data,
enable a GPU and Internet, and run all cells. Every random seed is fixed.

## Data and licence notes

The APTOS 2019 images are covered by the Kaggle competition rules and are not included in this repository.
The trained Keras file (`dr_model.keras`, 53 MB) is not included either; it is produced by the notebook.

## Disclaimer

Research prototype built for coursework. Not a medical device and not for diagnosis.
