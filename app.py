# app.py (TFLite-only, Render-friendly)
import os
import json
import numpy as np
from flask import Flask, request, render_template, send_from_directory, abort
from PIL import Image

# Try to import the lightweight TFLite runtime
try:
    import tflite_runtime.interpreter as tflite
except Exception as e:
    raise RuntimeError(
        "tflite_runtime is not available. Install tflite-runtime in requirements "
        "or use a runtime that provides it."
    ) from e

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Model & mapping files (must be present in repo)
TFLITE_MODEL = "arecanut_model.tflite"
CLASS_MAP = "class_indices.json"

if not os.path.exists(TFLITE_MODEL):
    raise FileNotFoundError(f"TFLite model not found at {TFLITE_MODEL}. Add it to the repo.")

if not os.path.exists(CLASS_MAP):
    raise FileNotFoundError(f"class_indices.json not found at {CLASS_MAP}. Add it to the repo.")

# Load class mapping (train_gen.class_indices style: {class_name: idx})
with open(CLASS_MAP, "r") as f:
    class_indices = json.load(f)
# invert mapping to idx -> class_name (ensure int keys)
idx_to_class = {int(v): k for k, v in class_indices.items()}

# Load TFLite interpreter
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper to preprocess image to match model input
def preprocess_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    # infer target_size from interpreter if not provided
    if target_size is None:
        shape = input_details[0]["shape"]
        # shape typically: [1, height, width, channels]
        if len(shape) == 4:
            target_size = (int(shape[2]), int(shape[1])) if False else (int(shape[1]), int(shape[2]))
            # choose (width, height) = (shape[2], shape[1])? We'll use (shape[1], shape[2]) as (H,W)
            target_size = (int(shape[2]), int(shape[1]))  # (W, H)
        else:
            target_size = (224, 224)
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0

    # Ensure shape is (1, H, W, C)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    # Cast to input dtype expected by model
    expected_dtype = np.dtype(input_details[0]["dtype"].name) if hasattr(input_details[0]["dtype"], "name") else input_details[0]["dtype"]
    arr = arr.astype(expected_dtype)
    return arr

def predict_tflite(image_path):
    inp = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(out))
    conf = float(out[idx])
    label = idx_to_class.get(idx, "unknown")
    return label, conf

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["POST"])
def upload_predict():
    if "file" not in request.files:
        return "No file part", 400
    f = request.files["file"]
    if f.filename == "":
        return "No selected file", 400

    # Save file
    safe_name = f.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    f.save(save_path)

    try:
        label, conf = predict_tflite(save_path)
    except Exception as e:
        # If something goes wrong with TFLite prediction, return 500 with message
        return f"Prediction failed: {str(e)}", 500

    return render_template("result.html", label=label, confidence=conf, filename=safe_name)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Use PORT from env for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
