"""
Flask web application for skateboarding trick classification
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Configuration
SEQUENCE_LENGTH = 40
LABELS = ["Back180", "Front180", "Frontshuvit", "Kickflip", "Ollie", "Shuvit", "Varial"]
ALLOWED_EXTENSIONS = {'mov', 'mp4', 'avi', 'mkv', 'webm'}

# Load the model (will be loaded on first request)
model = None
feature_extractor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load the trained model and feature extractor"""
    global model, feature_extractor
    
    if model is None:
        # Try to load the best model, fall back to final model
        if os.path.exists('best_model.keras'):
            model = tf.keras.models.load_model('best_model.keras')
            print("Loaded best_model.keras")
        elif os.path.exists('final_model.keras'):
            model = tf.keras.models.load_model('final_model.keras')
            print("Loaded final_model.keras")
        else:
            raise FileNotFoundError("No trained model found. Please run training first.")
    
    if feature_extractor is None:
        # Create feature extractor
        inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        x = inception_v3.output
        pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)
        feature_extractor = tf.keras.Model(inception_v3.input, pooling_output)
        print("Feature extractor loaded")

def extract_features_from_video(video_path):
    """Extract features from a video file"""
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every_frame = max(1, num_frames // SEQUENCE_LENGTH)
    
    features = []
    current_frame = 0
    frames_processed = 0
    
    while frames_processed < SEQUENCE_LENGTH:
        success, frame = cap.read()
        if not success:
            break
            
        if current_frame % sample_every_frame == 0:
            # Convert BGR to RGB
            frame = frame[:, :, ::-1]
            # Resize to 299x299
            img = tf.image.resize(frame, (299, 299))
            # Preprocess
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            # Extract features
            feature = feature_extractor(img)
            features.append(feature.numpy()[0])
            frames_processed += 1
            
        current_frame += 1
    
    cap.release()
    
    # Pad with zeros if needed
    padded_features = np.zeros((SEQUENCE_LENGTH, 2048))
    padded_features[:len(features)] = np.array(features)
    
    return padded_features

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', labels=LABELS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle video upload and prediction"""
    try:
        # Check if file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        # Load models if not already loaded
        load_models()
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract features from video
            features = extract_features_from_video(filepath)
            
            # Make prediction
            features_batch = np.expand_dims(features, axis=0)
            predictions = model.predict(features_batch, verbose=0)
            
            # Get results
            predicted_idx = np.argmax(predictions[0])
            predicted_label = LABELS[predicted_idx]
            confidence = float(predictions[0][predicted_idx])
            
            # Get all probabilities
            all_predictions = {
                LABELS[i]: float(predictions[0][i]) 
                for i in range(len(LABELS))
            }
            
            # Sort by confidence
            sorted_predictions = sorted(
                all_predictions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return jsonify({
                'success': True,
                'predicted_trick': predicted_label,
                'confidence': confidence,
                'all_predictions': sorted_predictions
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        load_models()
        return jsonify({'status': 'healthy', 'model_loaded': True})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("SkateboardML Web Application")
    print("="*50)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
