# app.py (TFLite-only, Render-friendly) — updated
import os
import json
import logging
import numpy as np
from flask import Flask, request, render_template, send_from_directory, abort, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arecanut-app")

# Try to import the lightweight TFLite runtime; fallback to tensorflow.lite if available
tflite = None
try:
    import tflite_runtime.interpreter as tflite
    logger.info("Using tflite_runtime.interpreter")
except Exception:
    try:
        import tensorflow as _tf  # optional fallback (bigger dependency)
        tflite = _tf.lite
        logger.info("tflite_runtime not available — falling back to tensorflow.lite")
    except Exception:
        raise RuntimeError(
            "Neither tflite_runtime nor tensorflow.lite is available. "
            "Install tflite-runtime in requirements or use a runtime that provides it."
        )

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Model & mapping files (must be present in repo)
TFLITE_MODEL = "arecanut_model.tflite"
CLASS_MAP = "class_indices.json"

# Basic existence checks
if not os.path.exists(TFLITE_MODEL):
    raise FileNotFoundError(f"TFLite model not found at {TFLITE_MODEL}. Add it to the repo.")

if not os.path.exists(CLASS_MAP):
    raise FileNotFoundError(f"class_indices.json not found at {CLASS_MAP}. Add it to the repo.")

# Load class mapping in a robust way:
with open(CLASS_MAP, "r") as f:
    class_indices_raw = json.load(f)

# class_indices may be either {"class_name": idx} or {"0": "class_name"} etc.
# Normalize to idx -> class_name (int keys)
idx_to_class = {}
if all(isinstance(k, str) and k.isdigit() for k in class_indices_raw.keys()):
    # keys are numeric strings -> map int(key) -> value (class name)
    for k, v in class_indices_raw.items():
        idx_to_class[int(k)] = v
else:
    # assume format {"class_name": idx}
    for k, v in class_indices_raw.items():
        try:
            idx = int(v)
            idx_to_class[idx] = k
        except Exception:
            # fallback: if values are strings and appear to be class names
            # try invert if values unique
            pass

if not idx_to_class:
    # final attempt: invert mapping (value->key)
    for k, v in class_indices_raw.items():
        try:
            idx = int(v)
            idx_to_class[idx] = k
        except Exception:
            # if values are indices encoded as strings use that
            pass

if not idx_to_class:
    raise ValueError("Unable to parse class_indices.json into an idx->class mapping.")

logger.info("Loaded class mapping: %s", idx_to_class)

# Load TFLite interpreter
try:
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite interpreter loaded. Input details: %s", input_details)
except Exception as e:
    logger.exception("Failed to load TFLite interpreter.")
    raise

# Helper: allowed file extension
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

# Helper to determine expected input shape & dtype
def _get_input_size_and_dtype():
    # Choose first input (most models have single input)
    inp = input_details[0]
    shape = inp.get("shape") or inp.get("shape_signature")
    # shape may be [1, H, W, C] or [-1, H, W, C] or [None, H, W, C]
    if shape is None or len(shape) < 4:
        target_h, target_w = 224, 224
    else:
        # take positions 1 and 2 as H, W
        try:
            target_h = int(shape[1]) if shape[1] not in (-1, None) else 224
            target_w = int(shape[2]) if shape[2] not in (-1, None) else 224
        except Exception:
            # fallback
            target_h, target_w = 224, 224
    # dtype may be numpy dtype object or string
    dtype = inp.get("dtype")
    # if dtype is a numpy type object, convert to np.dtype
    try:
        expected_dtype = np.dtype(dtype)
    except Exception:
        # if tflite returned something else, assume float32
        expected_dtype = np.float32
    return (target_w, target_h), expected_dtype  # PIL resize expects (W, H)

# Preprocess
def preprocess_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is None:
        target_size, expected_dtype = _get_input_size_and_dtype()
    else:
        _, expected_dtype = _get_input_size_and_dtype()

    # Resize (PIL expects (width, height))
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0

    # Ensure batch dim
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    # Cast to expected dtype if needed
    expected_dtype = expected_dtype if isinstance(expected_dtype, np.dtype) else np.dtype(expected_dtype)
    if arr.dtype != expected_dtype:
        arr = arr.astype(expected_dtype)

    # If model expects a different tensor order (rare), user can adjust here.
    return arr

# Prediction
def predict_tflite(image_path):
    inp = preprocess_image(image_path)
    try:
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])[0]
    except Exception as e:
        logger.exception("TFLite invocation error")
        raise

    idx = int(np.argmax(out))
    conf = float(out[idx])
    label = idx_to_class.get(idx, f"label_{idx}")
    return label, conf, out.tolist()

# Routes
@app.route("/", methods=["GET"])
def index_route():
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
    filename = secure_filename(f.filename)
    if not allowed_file(filename):
        return "Unsupported file type", 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)
    logger.info("Saved upload to %s", save_path)

    try:
        label, conf, raw_output = predict_tflite(save_path)
    except Exception as e:
        logger.exception("Prediction failed")
        return f"Prediction failed: {str(e)}", 500

    # Render result page if template exists, else return JSON
    try:
        return render_template("result.html", label=label, confidence=conf, filename=filename)
    except Exception:
        # fallback JSON
        return jsonify({"label": label, "confidence": conf, "raw_output": raw_output})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    logger.info("Starting app on %s:%s", host, port)
    app.run(host=host, port=port)
