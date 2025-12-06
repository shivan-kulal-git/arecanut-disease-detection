# (app.py content -- same as earlier I provided)
import os, json, numpy as np
from flask import Flask, request, render_template, send_from_directory
from PIL import Image

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open('class_indices.json','r') as f:
    class_indices = json.load(f)
idx_to_class = {int(v):k for k,v in class_indices.items()}

TFLITE_MODEL = 'arecanut_model.tflite'
KERAS_MODEL = 'arecanut_model.keras'

use_tflite = False
interpreter = None
if os.path.exists(TFLITE_MODEL):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(TFLITE_MODEL)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    use_tflite = True
else:
    from tensorflow.keras.models import load_model
    model = load_model(KERAS_MODEL)

def preprocess_image(path, target_size=(224,224)):
    img = Image.open(path).convert('RGB').resize(target_size)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def predict(path):
    arr = preprocess_image(path)
    if use_tflite:
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        out = model.predict(arr)[0]
    idx = int(np.argmax(out))
    confidence = float(out[idx])
    label = idx_to_class[idx]
    return label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return 'No file part', 400
    f = request.files['file']
    if f.filename == '':
        return 'No selected file', 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(filepath)
    label, conf = predict(filepath)
    return render_template('result.html', label=label, confidence=conf, filename=f.filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
