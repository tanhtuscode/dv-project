import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import config, SEQUENCE_LENGTH, LABELS, ALLOWED_EXTENSIONS

try:
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input
except ImportError:
    print("Warning: TensorFlow not found. Some features may not work.")

app = Flask(__name__)
app.secret_key = 'skateboardml_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Dynamic configuration that works on any PC
BASE_PATH = str(config.PROJECT_ROOT)
TRICKS_PATH = str(config.TRICKS_DIR)
UPLOAD_FOLDER = str(config.UPLOADS_DIR)
MODEL_PATH = str(config.MODELS_DIR)
TRAIN_LIST_PATH = str(config.TRAIN_LIST)
TEST_LIST_PATH = str(config.TEST_LIST)

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Load feature extraction model
feature_extractor = None
try:
    print("Loading InceptionV3 for feature extraction...")
    feature_extractor = InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(299, 299, 3)
    )
    print("InceptionV3 loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load InceptionV3: {e}")
    feature_extractor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features_from_video(video_path, max_frames=40):
    """Extract features from video using InceptionV3"""
    if feature_extractor is None:
        raise Exception("Feature extractor not available. Please ensure TensorFlow is properly installed.")
    
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame to 299x299 for InceptionV3
        frame = cv2.resize(frame, (299, 299))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = preprocess_input(frame)
        
        # Extract features
        feature = feature_extractor.predict(frame, verbose=0)
        features.append(feature[0])
        frame_count += 1
    
    cap.release()
    return np.array(features)

def load_best_model():
    """Load the best available model"""
    model_files = ['best_model.keras', 'final_model.keras']
    for model_file in model_files:
        model_path = os.path.join(MODEL_PATH, model_file)  # Fixed: use MODEL_PATH instead of BASE_PATH
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                return model, model_file
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
                continue
    return None, None

def update_training_files(video_filename, trick_label, split='train'):
    """Add new video to training or test lists"""
    if split == 'train':
        list_file = os.path.join(BASE_PATH, 'data', 'trainlist_binary.txt')
    else:
        list_file = os.path.join(BASE_PATH, 'data', 'testlist_binary.txt')
    
    # Create the entry for the list file
    entry = f"{trick_label}/{video_filename}"
    
    # Append to the appropriate list file
    with open(list_file, 'a') as f:
        f.write(f"\n{entry}")
    
    return True

@app.route('/')
def index():
    """Main page with navigation to all functions"""
    return render_template('index.html', labels=LABELS)

@app.route('/data_collection')
def data_collection():
    """Page for collecting new training data"""
    return render_template('data_collection.html', labels=LABELS)

@app.route('/model_testing')
def model_testing():
    """Page for manually testing the model"""
    return render_template('model_testing.html', labels=LABELS)

@app.route('/prediction')
def prediction():
    """Page for final prediction using best model"""
    return render_template('prediction.html', labels=LABELS)

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """Handle training data upload and labeling"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        trick_label = request.form.get('trick_label')
        split = request.form.get('split', 'train')  # train or test
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not trick_label:
            return jsonify({'error': 'No trick label provided'}), 400
        
        if trick_label not in LABELS:
            return jsonify({'error': f'Invalid trick label. Must be one of: {LABELS}'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded video
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{trick_label}_{timestamp}_{filename}"
            
            # Create trick directory if it doesn't exist
            trick_dir = os.path.join(TRICKS_PATH, trick_label)
            os.makedirs(trick_dir, exist_ok=True)
            
            video_path = os.path.join(trick_dir, video_filename)
            file.save(video_path)
            
            # Extract features and save as .npy
            features = extract_features_from_video(video_path)
            npy_filename = video_filename.rsplit('.', 1)[0] + '.npy'
            npy_path = os.path.join(trick_dir, npy_filename)
            np.save(npy_path, features)
            
            # Update training/test lists
            update_training_files(video_filename, trick_label, split)
            
            return jsonify({
                'success': True,
                'message': f'Video uploaded and processed successfully! Added to {split} set.',
                'video_path': video_path,
                'features_shape': features.shape,
                'split': split
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Error processing upload: {str(e)}'}), 500

@app.route('/test_model', methods=['POST'])
def test_model():
    """Test the model with user-uploaded video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        actual_label = request.form.get('actual_label', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save to temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)
            
            try:
                # Extract features
                features = extract_features_from_video(temp_path)
                
                # Load model
                model, model_name = load_best_model()
                if model is None:
                    return jsonify({'error': 'No trained model found. Please train a model first.'}), 400
                
                # Prepare features for prediction
                padded_sequence = np.zeros((1, SEQUENCE_LENGTH, 2048), dtype=np.float32)
                seq_len = min(len(features), SEQUENCE_LENGTH)
                padded_sequence[0, :seq_len] = features[:seq_len]
                
                # Make prediction
                predictions = model.predict(padded_sequence, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_label = LABELS[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])
                
                # Get all probabilities
                all_probabilities = {}
                for i, label in enumerate(LABELS):
                    all_probabilities[label] = float(predictions[0][i])
                
                result = {
                    'success': True,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'all_probabilities': all_probabilities,
                    'model_used': model_name,
                    'features_extracted': features.shape[0]
                }
                
                if actual_label:
                    result['actual_label'] = actual_label
                    result['correct'] = predicted_label == actual_label
                
                return jsonify(result)
                
            finally:
                # Clean up temporary files
                shutil.rmtree(temp_dir)
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Error testing model: {str(e)}'}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Final prediction using the best model"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save to temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)
            
            try:
                # Extract features
                features = extract_features_from_video(temp_path)
                
                # Load best model
                model, model_name = load_best_model()
                if model is None:
                    return jsonify({'error': 'No trained model found. Please train a model first.'}), 400
                
                # Prepare features for prediction
                padded_sequence = np.zeros((1, SEQUENCE_LENGTH, 2048), dtype=np.float32)
                seq_len = min(len(features), SEQUENCE_LENGTH)
                padded_sequence[0, :seq_len] = features[:seq_len]
                
                # Make prediction
                predictions = model.predict(padded_sequence, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_label = LABELS[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])
                
                # Get all probabilities
                all_probabilities = {}
                for i, label in enumerate(LABELS):
                    all_probabilities[label] = float(predictions[0][i])
                
                return jsonify({
                    'success': True,
                    'predicted_trick': predicted_label,
                    'confidence': confidence,
                    'confidence_percentage': f"{confidence * 100:.1f}%",
                    'all_probabilities': all_probabilities,
                    'model_used': model_name,
                    'features_extracted': features.shape[0],
                    'video_length_frames': features.shape[0]
                })
                
            finally:
                # Clean up temporary files
                shutil.rmtree(temp_dir)
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Trigger model training with current data"""
    try:
        # Import and run training
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from train import run_training  # Fixed: import from train instead of train_windows
        
        epochs = request.form.get('epochs', 10, type=int)
        
        # Run training in background (you might want to use threading for this)
        run_training(epochs=epochs)
        
        return jsonify({
            'success': True,
            'message': f'Training completed with {epochs} epochs!',
            'model_saved': 'best_model.keras and final_model.keras'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

@app.route('/get_model_info')
def get_model_info():
    """Get information about available models"""
    try:
        model_info = []
        model_files = ['best_model.keras', 'final_model.keras']
        
        for model_file in model_files:
            model_path = os.path.join(BASE_PATH, model_file)
            if os.path.exists(model_path):
                stat = os.stat(model_path)
                model_info.append({
                    'name': model_file,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return jsonify({
            'success': True,
            'models': model_info,
            'current_labels': LABELS
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting model info: {str(e)}'}), 500

@app.route('/get_dataset_stats')
def get_dataset_stats():
    """Get current dataset statistics"""
    try:
        stats = {}
        
        # Count files in each trick directory
        for label in LABELS:
            trick_dir = os.path.join(TRICKS_PATH, label)
            if os.path.exists(trick_dir):
                mov_files = len([f for f in os.listdir(trick_dir) if f.endswith('.mov')])
                npy_files = len([f for f in os.listdir(trick_dir) if f.endswith('.npy')])
                stats[label] = {'videos': mov_files, 'features': npy_files}
            else:
                stats[label] = {'videos': 0, 'features': 0}
        
        # Count entries in training lists
        train_count = 0
        test_count = 0
        
        train_file = os.path.join(BASE_PATH, 'data', 'trainlist_binary.txt')
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                train_count = len([line.strip() for line in f if line.strip()])
        
        test_file = os.path.join(BASE_PATH, 'data', 'testlist_binary.txt')
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                test_count = len([line.strip() for line in f if line.strip()])
        
        return jsonify({
            'success': True,
            'trick_stats': stats,
            'train_samples': train_count,
            'test_samples': test_count,
            'total_samples': train_count + test_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting dataset stats: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting SkateboardML Web Application...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Tricks path: {TRICKS_PATH}")
    print(f"Available labels: {LABELS}")
    app.run(debug=True, host='0.0.0.0', port=5000)