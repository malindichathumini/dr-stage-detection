"""
Convert the trained Keras model into the files the website needs.

  model/backbone.onnx : EfficientNetB0 feature extractor (image 0-255 -> 10x10x1280 feature map)
  model/head.bin      : float32 weights of the classification head (Dense 256 + Dense 5)

The head and Grad-CAM are computed in JavaScript (app.js), which reproduces the Keras
probabilities exactly. Usage (in the same Python environment as the local app):

  pip install tf2onnx onnx onnxruntime
  python tools/convert_to_onnx.py path/to/dr_model.keras
"""
import os
import subprocess
import sys

import keras
import numpy as np

model_path = sys.argv[1] if len(sys.argv) > 1 else "dr_model.keras"
os.makedirs("model", exist_ok=True)

model = keras.models.load_model(model_path)
base = next(layer for layer in model.layers if isinstance(layer, keras.Model))   # the EfficientNet backbone

# 1. Backbone -> SavedModel -> ONNX
backbone = keras.Model(base.inputs, base.outputs)
backbone.export("backbone_savedmodel", format="tf_saved_model")
subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", "backbone_savedmodel",
                "--output", "model/backbone.onnx", "--opset", "17"], check=True)

# 2. Head weights, concatenated as float32: W1 (1280x256), b1, W2 (256x5), b2
W1, b1 = model.get_layer("fc").get_weights()
W2, b2 = model.get_layer("stage").get_weights()
np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype(np.float32).tofile("model/head.bin")
print("Wrote model/backbone.onnx and model/head.bin")
