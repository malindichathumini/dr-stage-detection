"""
Convert one trained Keras model (an ensemble member) into the files the website needs.

  <out>/backbone.onnx  or  <out>/backbone.partNN.onnx   EfficientNet feature extractor, image (0-255) -> feature map
                                                       (split into parts under 20 MB for GitHub web upload)
  <out>/head.bin                                       float32 head weights: W1, b1 (Dense 256), W2, b2 (Dense 5)

The head and Grad-CAM are computed in JavaScript (app.js), which reproduces the Keras probabilities.
Add the member to model/config.json (name, img_size, onnx parts, head, head_dims, feature_shape).

Usage (in a Python environment with TensorFlow/Keras):
  pip install tf2onnx onnx
  python tools/convert_to_onnx.py dr_model_member1.keras model/b3
"""
import json
import os
import subprocess
import sys

import keras
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "dr_model.keras"
out = sys.argv[2] if len(sys.argv) > 2 else "model"
os.makedirs(out, exist_ok=True)
PART = 19 * 1000 * 1000                                          # keep every file under GitHub's web limit

model = keras.models.load_model(src, compile=False)
base = next(layer for layer in model.layers if isinstance(layer, keras.Model))   # the EfficientNet backbone

# 1. Backbone -> SavedModel -> ONNX
backbone = keras.Model(base.inputs, base.outputs)
backbone.export("backbone_savedmodel", format="tf_saved_model")
onnx_file = os.path.join(out, "backbone.onnx")
subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", "backbone_savedmodel",
                "--output", onnx_file, "--opset", "17"], check=True)

# 2. Split large files into parts (joined again in the browser)
data = open(onnx_file, "rb").read()
if len(data) > PART:
    os.remove(onnx_file)
    parts = []
    for i in range(0, len(data), PART):
        name = f"backbone.part{i // PART:02d}.onnx"
        open(os.path.join(out, name), "wb").write(data[i:i + PART])
        parts.append(f"{out}/{name}")
else:
    parts = [f"{out}/backbone.onnx"]

# 3. Head weights, concatenated as float32
W1, b1 = model.get_layer("fc").get_weights()
W2, b2 = model.get_layer("stage").get_weights()
np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype(np.float32).tofile(os.path.join(out, "head.bin"))

# 4. The entry to add to model/config.json -> "members"
print(json.dumps({"img_size": int(model.input_shape[1]), "onnx": parts, "head": f"{out}/head.bin",
                  "head_dims": {"in": int(W1.shape[0]), "hidden": int(W1.shape[1]), "out": int(W2.shape[1])},
                  "feature_shape": [int(d) for d in backbone.output_shape[1:]]}, indent=1))
