# Diabetic Retinopathy Stage Detection

An ensemble of three fine-tuned convolutional neural networks (EfficientNetB0, B3 and B4) that grades retinal
fundus photographs on the international 5-stage diabetic retinopathy (DR) scale, with Grad-CAM explanations
and a web prototype that runs entirely in the browser.

## Links

| What | Link |
|---|---|
| Live website (runs in your browser) | https://dr-stage-detection.vercel.app |
| Code (this repository) | https://github.com/malindichathumini/dr-stage-detection |
| **Final notebook: Version 4 ensemble, 80.4% (Kaggle)** | https://www.kaggle.com/code/malindichathumini/dr-stage-detection-ensemble-v4 |
| Experiments notebook: backbone, hyperparameter, preprocessing and balancing comparisons (Kaggle) | https://www.kaggle.com/code/malindichathumini/dr-stage-detection-experiments |
| Video demonstration | https://YOUR-VIDEO-LINK |

NIBM · BSc (Hons) Computer Science · Computer Vision · Coursework 1

## Results (held-out test set, 550 images)

Final model (Version 4): ensemble of EfficientNetB0 (300 px), EfficientNetB3 (380 px) and EfficientNetB4 (380 px).

| Metric | Value |
|---|---|
| Stage accuracy | **80.4%** (442 / 550) |
| Macro F1 | 0.644 |
| Quadratic weighted kappa (QWK) | 0.877 |
| DR vs No DR sensitivity / specificity | 97.8% / 97.8% |
| DR vs No DR ROC AUC | 0.996 |

| Stage | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| 0 · No DR | 0.971 | 0.978 | 0.974 | 271 |
| 1 · Mild | 0.487 | 0.679 | 0.567 | 56 |
| 2 · Moderate | 0.822 | 0.707 | 0.760 | 150 |
| 3 · Severe | 0.400 | 0.483 | 0.438 | 29 |
| 4 · Proliferative | 0.543 | 0.432 | 0.481 | 44 |

### How the results developed

| Version | What changed | Accuracy | Macro F1 | QWK |
|---|---|---|---|---|
| V1 | EfficientNetB0, 224 px, top 30% fine-tuned, loss-based model selection | 74.4% | 0.564 | 0.826 |
| V2 | 300 px, full fine-tuning (BatchNorm frozen), QWK-based early stopping, oversampling | 78.9% | 0.632 | 0.879 |
| V3 | Ensemble of B0 + B3 (300 px) | 79.8% | 0.654 | 0.888 |
| **V4** | **Ensemble of B0 (300 px) + B3 and B4 (380 px)** | **80.4%** | 0.644 | 0.877 |

Every choice (backbone, preprocessing, balancing, inference method, ensemble members) was made on the
validation set; the test set was used once per version. V4 has the best accuracy and Mild recall; V3 has a
slightly higher macro F1 and QWK, so the gain from V3 to V4 is a trade-off rather than a strict improvement.

## Technologies used

| Area | Tools |
|---|---|
| Model training | Python, TensorFlow / Keras (EfficientNetB0/B3/B4, ResNet50V2, MobileNetV2), scikit-learn, OpenCV, NumPy, pandas, Matplotlib (Kaggle, NVIDIA T4 GPU) |
| Web prototype | HTML, CSS, JavaScript, ONNX Runtime Web (WebAssembly), hosted on Vercel |
| Desktop prototype | Python, Gradio |
| Model conversion | tf2onnx, ONNX |

## Repository layout

| Path | What it is |
|---|---|
| `index.html`, `style.css`, `app.js` | The web prototype (static site, deployed on Vercel) |
| `model/backbone.onnx`, `model/head.bin` | Ensemble member 1: EfficientNetB0 (300 px) backbone in ONNX (16 MB) and head weights |
| `model/b3/` | Ensemble member 2: EfficientNetB3 (380 px), ONNX in 3 parts under 20 MB + head weights |
| `model/b4/` | Ensemble member 3: EfficientNetB4 (380 px), ONNX in 4 parts under 20 MB + head weights |
| `model/config.json` | Ensemble members, input sizes, class names, decision rule and test metrics |
| `vendor/` | ONNX Runtime Web 1.22 (WebAssembly), self-hosted |
| `assets/figures/` | Evaluation figures shown on the site |
| `notebook/DR_Stage_Detection.ipynb` | Version 2 pipeline: experiments, training and evaluation (Kaggle, GPU) |
| `notebook/DR_Stage_Detection_v4.ipynb` | Version 4: trains the B3 and B4 members at 380 px and selects the ensemble |
| `local_app/` | Gradio desktop version of the prototype |
| `tools/convert_to_onnx.py` | Converts one trained Keras model into the web files (ONNX parts + head) |

## Pipeline

1. **Data:** APTOS 2019 Blindness Detection (Kaggle), 3,662 images graded 0–4; stratified 70 / 15 / 15 split made before augmentation.
2. **Preprocessing:** black-border crop, pad to square, 300×300 area resize. CLAHE + unsharp masking and Ben Graham filtering were implemented and compared with full training; raw images scored best (validation QWK 0.861 vs 0.858 and 0.829).
3. **Augmentation:** flips, 360° rotation, zoom, translation, brightness and contrast (training only).
4. **Class balancing:** balanced weights, square-root weights and oversampling compared; oversampling selected.
5. **Model:** ImageNet EfficientNetB0 (selected against EfficientNetB3, ResNet50V2, MobileNetV2) + GAP → Dropout → Dense 256 (L2) → Dropout → Softmax 5. Version 4 adds EfficientNetB3 and EfficientNetB4 members trained the same way at 380 px (different seeds).
6. **Training:** phase 1 trains the head with the backbone frozen; phase 2 fine-tunes the whole backbone (BatchNorm frozen) at 2e-5. Early stopping and checkpointing on validation QWK; learning rate reduced on plateau; label smoothing 0.05.
7. **Inference:** TTA and QWK-optimised thresholds were evaluated; plain argmax was kept because it gave the best balance of accuracy, macro F1 and QWK on the validation set (thresholds cut Mild recall to 12.5%).
8. **Ensemble:** every combination of the trained models, with and without TTA, was scored on the validation set (mean of accuracy, macro F1 and QWK); the best was B0 + B3-380 + B4-380 without TTA (validation accuracy 81.2%). The members' probabilities are averaged.
9. **Explainability:** Grad-CAM on the last convolutional feature map of each member, averaged over the ensemble.

## How the web version works

Each Keras model in the ensemble is split in two. Its EfficientNet backbone is exported to ONNX and runs in
the browser with ONNX Runtime Web (WebAssembly); the larger backbones are stored in parts under 20 MB and
joined after download. The small classification heads and Grad-CAM are computed in `app.js` from the
exported weights, with Grad-CAM gradients derived analytically through the dense layers. Each model gets the
image at its own input size and the probabilities are averaged. On the example images the browser matches the
Keras ensemble to within 0.001 and always gives the same stage; one analysis takes about 2 seconds on a laptop.
The first visit downloads about 130 MB of models, which the browser then caches. Because inference happens on the user's
device, the retinal image is never uploaded, which protects patient privacy and needs no paid server.

## Run it

**Website locally:** any static server, for example `python -m http.server 8000`, then open http://localhost:8000.

**Desktop app:** see `local_app/`. Put `dr_config.json`, the model files it lists (`dr_model.keras`,
`dr_model_member1.keras`, `dr_model_member2.keras`), `results.json` and `classification_report.csv` from the
Version 4 notebook output next to `app.py`, install `requirements.txt`, then run
`python app.py` (or `start_app.bat` on Windows).

**Retrain:** import `notebook/DR_Stage_Detection.ipynb` into Kaggle, attach the APTOS 2019 competition data,
enable a GPU and Internet, and run all cells. Then run `notebook/DR_Stage_Detection_v4.ipynb` with the
Version 2/3 notebook output attached as input to train the 380 px members and build the ensemble. Every random
seed is fixed.

## Data and licence notes

The APTOS 2019 images are covered by the Kaggle competition rules and are not included in this repository.
The trained Keras files (53 MB, 134 MB and 217 MB) are not included either; they are produced by the notebooks.

## Disclaimer

Research prototype built for coursework. Not a medical device and not for diagnosis.
