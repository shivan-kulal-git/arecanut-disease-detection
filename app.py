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
    """
    Robustly return (target_size (W,H), expected_dtype).
    Handles numpy scalars / shape signatures safely.
    """
    inp = input_details[0]
    # prefer shape_signature if available (may contain -1)
    shape = inp.get("shape_signature") or inp.get("shape")
    # defensive: convert to plain Python ints where possible
    try:
        # shape may be a list/tuple of numpy scalars; convert element-wise to int if possible
        shape_list = [int(x) for x in shape]
    except Exception:
        # fallback defaults
        shape_list = [-1, 224, 224, 3]

    # Expect shape like [batch, H, W, C]
    if len(shape_list) >= 4:
        h = shape_list[1]
        w = shape_list[2]
        # if either is <=0 (unknown), fallback to 224
        target_h = h if (isinstance(h, int) and h > 0) else 224
        target_w = w if (isinstance(w, int) and w > 0) else 224
    else:
        target_w, target_h = 224, 224

    # dtype: try to convert to numpy dtype
    dtype = inp.get("dtype", np.float32)
    try:
        expected_dtype = np.dtype(dtype)
    except Exception:
        expected_dtype = np.float32

    # Return PIL-style size (W, H) and dtype
    return (int(target_w), int(target_h)), expected_dtype


def preprocess_image(path, target_size=None):
    """
    Load image, resize to (W,H), convert to float (0..1) and cast to expected dtype.
    Always returns a batched array: shape (1, H, W, C)
    """
    img = Image.open(path).convert("RGB")

    # get expected size + dtype if not provided
    if target_size is None:
        target_size, expected_dtype = _get_input_size_and_dtype()
    else:
        _, expected_dtype = _get_input_size_and_dtype()

    # PIL resize expects (width, height)
    img = img.resize(target_size, Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0

    # Ensure batch dim (1, H, W, C)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    # Final dtype cast to what the model expects
    if not isinstance(expected_dtype, np.dtype):
        expected_dtype = np.dtype(expected_dtype)
    # if arrays mismatched (e.g. model expects uint8), cast now
    if arr.dtype != expected_dtype:
        try:
            arr = arr.astype(expected_dtype)
        except Exception:
            # as a safe fallback use float32
            arr = arr.astype(np.float32)

    return arr


def predict_tflite(image_path):
    """
    Calls the TFLite interpreter in a defensive way and returns (label, confidence, raw_out).
    """
    inp = preprocess_image(image_path)
    # sanity-check shapes & types
    if inp.ndim != 4:
        raise ValueError(f"Preprocessed input must be 4-D (B,H,W,C). Got shape {inp.shape}")

    # Set tensor and invoke
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    raw_out = interpreter.get_tensor(output_details[0]["index"])
    # Make sure raw_out is an array and has the expected final dimension
    raw_out = np.asarray(raw_out)
    if raw_out.ndim == 2 and raw_out.shape[0] == 1:
        out_vec = raw_out[0]
    elif raw_out.ndim == 1:
        out_vec = raw_out
    else:
        # try to flatten last dimension
        out_vec = raw_out.reshape(-1, raw_out.shape[-1])[0]

    # safe argmax & cast
    idx = int(np.argmax(out_vec))
    conf = float(out_vec[idx]) if out_vec.size > idx else float(np.max(out_vec))
    label = idx_to_class.get(idx, f"label_{idx}")
    return label, conf, out_vec.tolist()


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
