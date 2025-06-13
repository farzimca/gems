# app.py

from flask import Flask, request, render_template
import pickle
import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
import os
from werkzeug.utils import secure_filename

# Load model
with open('modelx.pkl', 'rb') as f:
    model = pickle.load(f)

#class name mapping
class_names = {
    0: 'amethyst',
    1: 'ametrine',
    2: 'black_spinel',
    3: 'cat_eye',
    4: 'citrine',
    5: 'diamond',
    6: 'emerald',
    7: 'pearl',
    8: 'peridot',
    9: 'rose_quartz',
    10: 'ruby',
    11: 'sapphire_blue',
    12: 'sapphire_yellow',
    13: 'smoky_quartz',
    14: 'turquoise',
    15: 'Unknown'
}

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (128, 128)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Feature Extraction Functions ---
def extract_color_features(hsv_img):
    feats = []
    for i in range(3):
        channel = hsv_img[:, :, i].flatten()
        feats.extend([
            np.mean(channel),
            np.std(channel),
            np.var(channel),
            skew(channel),
            kurtosis(channel)
        ])
    return feats

def extract_texture_features(gray_img):
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return [contrast, energy, homogeneity, correlation]

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMAGE_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_feats = extract_color_features(hsv)
    texture_feats = extract_texture_features(gray)
    return np.array(color_feats + texture_feats).reshape(1, -1)

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No selected file.")

    if file and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract features from image
        features = process_image(filepath)

        # Check if the model supports predict_proba (probabilities of classes)
        if hasattr(model, "predict_proba"):
            # Get the probabilities for each class
            probs = model.predict_proba(features)
            
            # Get the maximum probability and predicted class
            max_prob = np.max(probs)
            predicted_class = np.argmax(probs)
            
            # If confidence is below a threshold, classify as Unknown
            threshold = 0.6  # You can adjust this threshold
            if max_prob < threshold:
                predicted_label = "Unknown Gemstone"
            else:
                predicted_label = class_names.get(predicted_class, f"Unknown class: {predicted_class}")
        else:
            # Fallback: If predict_proba isn't available, just use predict
            prediction = model.predict(features)
            predicted_class = prediction[0]
            predicted_label = class_names.get(predicted_class, f"Unknown class: {predicted_class}")

        return render_template('index.html', prediction_text=f'Predicted Class: {predicted_label}', image_path=filepath)

    return render_template('index.html', prediction_text="Invalid file format.")


if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
