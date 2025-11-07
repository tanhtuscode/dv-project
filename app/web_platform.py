import os
import shutil
import tempfile
import json
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import sys
from collections import defaultdict, Counter

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import config, SEQUENCE_LENGTH, ALLOWED_EXTENSIONS

try:
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input
except ImportError:
    print("Warning: TensorFlow not found. Some features may not work.")

app = Flask(__name__)
app.secret_key = 'skateboardml_platform_2024'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

# Dynamic configuration
TRICKS_PATH = str(config.TRICKS_DIR)
UPLOAD_FOLDER = str(config.UPLOADS_DIR)
MODEL_PATH = str(config.MODELS_DIR)
DATA_PATH = str(config.DATA_DIR)

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(TRICKS_PATH, exist_ok=True)

# Global variables
feature_extractor = None
current_labels = []
loaded_model = None
model_info = {}

# Initialize feature extractor
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

def get_available_labels():
    """Get all available labels from the tricks directory"""
    global current_labels
    if not current_labels:
        tricks_dir = Path(TRICKS_PATH)
        if tricks_dir.exists():
            current_labels = [d.name for d in tricks_dir.iterdir() if d.is_dir()]
        if not current_labels:
            current_labels = ['Kickflip', 'Ollie']
    return current_labels

def update_data_lists():
    """Automatically update train/test/validation lists"""
    tricks_dir = Path(TRICKS_PATH)
    all_files = []
    
    for label_dir in tricks_dir.iterdir():
        if label_dir.is_dir():
            for video_file in label_dir.glob("*.MOV"):
                npy_file = video_file.with_suffix('.npy')
                if npy_file.exists():
                    relative_path = str(video_file.relative_to(tricks_dir)).replace('\\', '/')
                    all_files.append(relative_path)
    
    # Split data: 70% train, 15% test, 15% validation
    np.random.shuffle(all_files)
    total = len(all_files)
    train_size = int(0.7 * total)
    test_size = int(0.15 * total)
    
    train_files = all_files[:train_size]
    test_files = all_files[train_size:train_size + test_size]
    val_files = all_files[train_size + test_size:]
    
    # Update files ONLY in data directory (single source of truth)
    data_dir = Path(DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / 'trainlist.txt', 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")
    
    with open(data_dir / 'testlist.txt', 'w') as f:
        for file in test_files:
            f.write(f"{file}\n")
    
    with open(data_dir / 'validationlist.txt', 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")
    
    print(f"[DATA LISTS] Updated: {len(train_files)} train, {len(test_files)} test, {len(val_files)} validation")
    
    return len(train_files), len(test_files), len(val_files)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_video_features(video_path):
    """Extract features from video using InceptionV3"""
    if feature_extractor is None:
        raise ValueError("Feature extractor not loaded")
    
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0
    max_frames = 100
    
    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (299, 299))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_input(frame.astype(np.float32))
            
            frame_features = feature_extractor.predict(np.expand_dims(frame, axis=0), verbose=0)
            features.append(frame_features[0])
            frame_count += 1
            
    finally:
        cap.release()
    
    return np.array(features)

def load_model_from_path(model_path):
    """Load a model from the given path"""
    global loaded_model, model_info
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        model_info = {
            'path': model_path,
            'name': os.path.basename(model_path),
            'loaded_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_shape': str(loaded_model.input_shape),
            'output_shape': str(loaded_model.output_shape),
            'total_params': loaded_model.count_params()
        }
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_with_loaded_model(features):
    """Make prediction using the loaded model"""
    if loaded_model is None:
        raise ValueError("No model loaded")
    
    padded_sequence = np.zeros((1, SEQUENCE_LENGTH, 2048), dtype=np.float32)
    seq_len = min(len(features), SEQUENCE_LENGTH)
    padded_sequence[0, :seq_len] = features[:seq_len]
    
    predictions = loaded_model.predict(padded_sequence, verbose=0)
    return predictions[0]

@app.route('/')
def index():
    """Main dashboard with all functionality"""
    labels = get_available_labels()
    
    # Auto-update data lists on page load
    train_count, test_count, val_count = 0, 0, 0
    try:
        train_count, test_count, val_count = update_data_lists()
        print(f"[AUTO-UPDATE] Data lists updated: {train_count} train, {test_count} test, {val_count} validation")
    except Exception as e:
        print(f"[WARNING] Could not update data lists: {e}")
        pass
    
    # Get available models
    available_models = []
    models_dir = Path(MODEL_PATH)
    if models_dir.exists():
        for model_file in models_dir.glob("*.keras"):
            available_models.append({
                'name': model_file.name,
                'path': str(model_file),
                'size': model_file.stat().st_size,
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Get label statistics
    label_stats = {}
    tricks_dir = Path(TRICKS_PATH)
    
    for label in labels:
        label_dir = tricks_dir / label
        if label_dir.exists():
            mov_files = list(label_dir.glob("*.MOV"))
            npy_files = list(label_dir.glob("*.npy"))
            label_stats[label] = {
                'videos': len(mov_files),
                'features': len(npy_files),
                'processed': len([f for f in mov_files if (f.parent / f.with_suffix('.npy').name).exists()])
            }
        else:
            label_stats[label] = {'videos': 0, 'features': 0, 'processed': 0}
    
    stats = {
        'total_labels': len(labels),
        'train_samples': train_count,
        'test_samples': test_count,
        'validation_samples': val_count,
        'total_samples': train_count + test_count + val_count
    }
    
    return render_template('index.html', 
                         labels=labels, 
                         stats=stats, 
                         model_info=model_info,
                         available_models=available_models,
                         label_stats=label_stats)

@app.route('/upload_multiple_videos', methods=['POST'])
def upload_multiple_videos():
    """Handle multiple video uploads with batch labeling"""
    if 'videos' not in request.files:
        return jsonify({'success': False, 'message': 'No videos selected'})
    
    files = request.files.getlist('videos')
    selected_label = request.form.get('label')
    create_new_label = request.form.get('create_new_label', '').strip()
    
    # Determine the label to use
    if create_new_label:
        label = create_new_label
        if label not in current_labels:
            current_labels.append(label)
    else:
        label = selected_label
    
    if not label:
        return jsonify({'success': False, 'message': 'No label specified'})
    
    # Create label directory
    label_dir = Path(TRICKS_PATH) / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    processed_count = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Generate unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{label}_{timestamp}_{processed_count:03d}.MOV"
                video_path = label_dir / filename
                
                # Save video
                file.save(str(video_path))
                
                # Extract features
                if feature_extractor:
                    features = extract_video_features(str(video_path))
                    npy_path = video_path.with_suffix('.npy')
                    np.save(str(npy_path), features)
                    
                    results.append({
                        'filename': filename,
                        'label': label,
                        'features_extracted': True,
                        'features_count': len(features)
                    })
                else:
                    results.append({
                        'filename': filename,
                        'label': label,
                        'features_extracted': False,
                        'error': 'Feature extractor not available'
                    })
                
                processed_count += 1
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
    
    # Update data lists
    try:
        train_count, test_count, val_count = update_data_lists()
        message = f"Successfully processed {processed_count} videos for label '{label}'. Data lists updated: {train_count} train, {test_count} test, {val_count} validation."
    except Exception as e:
        message = f"Successfully processed {processed_count} videos for label '{label}'. Warning: Could not update data lists: {e}"
    
    return jsonify({
        'success': True,
        'message': message,
        'results': results,
        'processed_count': processed_count,
        'label': label
    })

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """Handle model upload"""
    if 'model' not in request.files:
        return jsonify({'success': False, 'message': 'No model file selected'})
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No model file selected'})
    
    if file and file.filename.endswith(('.keras', '.h5')):
        try:
            # Save uploaded model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"uploaded_model_{timestamp}.keras"
            model_path = Path(MODEL_PATH) / filename
            file.save(str(model_path))
            
            # Try to load the model
            if load_model_from_path(str(model_path)):
                return jsonify({
                    'success': True,
                    'message': f'Model uploaded and loaded successfully: {filename}',
                    'model_info': model_info
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Model uploaded but failed to load. Please check model format.'
                })
                
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error uploading model: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid model file format. Use .keras or .h5 files.'})

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load an existing model"""
    model_path = request.form.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'success': False, 'message': 'Model file not found'})
    
    if load_model_from_path(model_path):
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {os.path.basename(model_path)}',
            'model_info': model_info
        })
    else:
        return jsonify({'success': False, 'message': 'Failed to load model'})

@app.route('/test_model_with_video', methods=['POST'])
def test_model_with_video():
    """Test loaded model with uploaded video"""
    if loaded_model is None:
        return jsonify({'success': False, 'message': 'No model loaded. Please load a model first.'})
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No video file selected'})
    
    file = request.files['video']
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid video file'})
    
    try:
        # Save temporary video
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_video_path)
        
        # Extract features
        features = extract_video_features(temp_video_path)
        
        # Make prediction
        predictions = predict_with_loaded_model(features)
        
        # Get current labels for interpretation
        labels = get_available_labels()
        
        # Prepare results
        results = []
        for i, prob in enumerate(predictions):
            if i < len(labels):
                results.append({
                    'label': labels[i],
                    'probability': float(prob),
                    'percentage': f"{prob * 100:.2f}%"
                })
        
        # Find predicted class
        predicted_idx = np.argmax(predictions)
        predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else f"Class_{predicted_idx}"
        confidence = float(predictions[predicted_idx])
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'all_predictions': results,
            'features_extracted': len(features)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error testing model: {str(e)}'})

@app.route('/batch_test_videos', methods=['POST'])
def batch_test_videos():
    """Test multiple videos with loaded model"""
    if loaded_model is None:
        return jsonify({'success': False, 'message': 'No model loaded. Please load a model first.'})
    
    if 'videos' not in request.files:
        return jsonify({'success': False, 'message': 'No videos selected'})
    
    files = request.files.getlist('videos')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Save temporary video
                temp_dir = tempfile.mkdtemp()
                temp_video_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(temp_video_path)
                
                # Extract features and predict
                features = extract_video_features(temp_video_path)
                predictions = predict_with_loaded_model(features)
                
                # Get labels and results
                labels = get_available_labels()
                predicted_idx = np.argmax(predictions)
                predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else f"Class_{predicted_idx}"
                confidence = float(predictions[predicted_idx])
                
                results.append({
                    'filename': file.filename,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'confidence_percentage': f"{confidence * 100:.2f}%",
                    'all_predictions': [
                        {'label': labels[i] if i < len(labels) else f"Class_{i}", 
                         'probability': float(predictions[i])}
                        for i in range(len(predictions))
                    ]
                })
                
                # Cleanup
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
    
    return jsonify({
        'success': True,
        'message': f'Tested {len(results)} videos',
        'results': results
    })

@app.route('/create_new_label', methods=['POST'])
def create_new_label():
    """Create a new label category"""
    label_name = request.form.get('label_name', '').strip()
    
    if not label_name:
        return jsonify({'success': False, 'message': 'Label name is required'})
    
    if label_name in get_available_labels():
        return jsonify({'success': False, 'message': 'Label already exists'})
    
    try:
        # Create directory for new label
        label_dir = Path(TRICKS_PATH) / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Add to current labels
        current_labels.append(label_name)
        
        # Auto-update data lists after creating new label
        try:
            train_count, test_count, val_count = update_data_lists()
            print(f"[AUTO-UPDATE] Data lists updated after creating label '{label_name}': {train_count} train, {test_count} test, {val_count} validation")
        except Exception as e:
            print(f"[WARNING] Could not update data lists after creating label: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Label "{label_name}" created successfully',
            'label': label_name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error creating label: {str(e)}'})

@app.route('/regenerate_data_lists', methods=['POST'])
def regenerate_data_lists():
    """Regenerate train/test/validation lists"""
    try:
        train_count, test_count, val_count = update_data_lists()
        
        return jsonify({
            'success': True,
            'message': 'Data lists regenerated successfully',
            'train_count': train_count,
            'test_count': test_count,
            'validation_count': val_count,
            'total_count': train_count + test_count + val_count
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error regenerating data lists: {str(e)}'})

@app.route('/download_data_lists')
def download_data_lists():
    """Download all data lists as a zip file"""
    try:
        # Create temporary zip file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'data_lists.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            data_dir = Path(DATA_PATH)
            for list_file in ['trainlist.txt', 'testlist.txt', 'validationlist.txt']:
                file_path = data_dir / list_file
                if file_path.exists():
                    zipf.write(str(file_path), list_file)
        
        return send_file(zip_path, as_attachment=True, download_name='data_lists.zip')
        
    except Exception as e:
        flash(f'Error creating download: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/get_label_statistics')
def get_label_statistics():
    """Get detailed statistics for all labels"""
    labels = get_available_labels()
    statistics = {}
    
    tricks_dir = Path(TRICKS_PATH)
    
    for label in labels:
        label_dir = tricks_dir / label
        if label_dir.exists():
            mov_files = list(label_dir.glob("*.MOV"))
            npy_files = list(label_dir.glob("*.npy"))
            
            # Calculate file sizes
            total_video_size = sum(f.stat().st_size for f in mov_files) / (1024 * 1024)  # MB
            total_feature_size = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)  # MB
            
            statistics[label] = {
                'video_count': len(mov_files),
                'feature_count': len(npy_files),
                'processed_count': len([f for f in mov_files if (f.parent / f.with_suffix('.npy').name).exists()]),
                'video_size_mb': round(total_video_size, 2),
                'feature_size_mb': round(total_feature_size, 2),
                'last_modified': max([f.stat().st_mtime for f in mov_files], default=0)
            }
        else:
            statistics[label] = {
                'video_count': 0,
                'feature_count': 0,
                'processed_count': 0,
                'video_size_mb': 0,
                'feature_size_mb': 0,
                'last_modified': 0
            }
    
    return jsonify({'success': True, 'statistics': statistics})

if __name__ == '__main__':
    # Initialize labels on startup
    get_available_labels()
    
    print("\n" + "="*60)
    print("SkateboardML Professional Platform")
    print("="*60)
    print(f"Available Labels: {get_available_labels()}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print(f"Tricks Path: {TRICKS_PATH}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Feature Extractor: {'Loaded' if feature_extractor else 'Not Available'}")
    print("="*60)
    print("Features:")
    print("1. Multi-video upload with batch labeling")
    print("2. Dynamic label management")
    print("3. Model upload and testing")
    print("4. Automatic data list generation")
    print("5. Data management and statistics")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)