import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import joblib
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array

# Flask setup
tmpl_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, template_folder=tmpl_dir, static_folder=static_dir)

# Configuration
UPLOAD_FOLDER = os.path.join(static_dir, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model artifacts at startup
feature_extractor = load_model('models/feature_extractor.h5')
pca = joblib.load('models/pca.pkl')
svm_model = joblib.load('models/svm_model.pkl')
has_proba = hasattr(svm_model, 'predict_proba')

# Category labels (adjust to your mapping)
categories = ['NonDemented', 'MildDemented', 'ModerateDemented', 'VeryMildDemented']

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run through feature extractor
    img_tensor = preprocess_image(filepath)
    features = feature_extractor.predict(img_tensor)

    # PCA reduction and SVM classification
    features_pca = pca.transform(features)
    pred_idx = int(svm_model.predict(features_pca)[0])
    stage = categories[pred_idx]

    # Confidence (if available)
    if has_proba:
        probs = svm_model.predict_proba(features_pca)[0]
        confidence = f"{np.max(probs) * 100:.1f}%"
    else:
        confidence = "N/A"

    # Description placeholder
    description = f"Predicted stage: {stage}."

    # Build response
    response = {
        'stage': stage,
        'confidence': confidence,
        'description': description,
        'image_url': url_for('static', filename=f'uploads/{filename}')
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
