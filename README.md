# SkateboardML - Skateboard Trick Classification

Professional ML platform for classifying skateboard tricks using CNN-LSTM architecture.

## Features

- **Multi-Video Upload**: Batch upload and label videos
- **Auto Feature Extraction**: Automatic InceptionV3 feature extraction
- **Model Training & Testing**: Upload and test custom models
- **Dynamic Labels**: Create new trick categories on-the-fly
- **Auto-Update Lists**: Automatic train/test/validation split generation
- **Professional UI**: Modern Bootstrap 5 interface

## Quick Start

### 1. Setup (First Time Only)

```bash
python setup.py
```

This will automatically:
- Create all required directories
- Install Python dependencies
- Generate configuration files
- Initialize data structures
- Create startup scripts

### 2. Run the Web Platform

**Windows:**
```bash
start_web_app.bat
```

**Linux/Mac:**
```bash
./start_web_app.sh
```

Or manually:
```bash
cd app
python web_platform.py
```

Then open: http://localhost:5000

## Project Structure

```
SkateboardML/
├── app/
│   ├── web_platform.py        # Main web application
│   ├── templates/             # HTML templates
│   ├── static/                # CSS, JS, images
│   └── uploads/               # Temporary uploads
├── data/
│   ├── Tricks/                # Video datasets
│   │   ├── Kickflip/
│   │   ├── Ollie/
│   │   └── ...
│   ├── trainlist.txt          # Training data list
│   ├── testlist.txt           # Testing data list
│   └── validationlist.txt     # Validation data list
├── models/                     # Trained models (.keras)
├── outputs/                    # Generated charts/visualizations
├── scripts/                    # Training and utility scripts
└── config/                     # Configuration files

```

## Usage

### Upload Videos

1. Navigate to the web platform
2. Select multiple videos
3. Choose or create a label
4. Upload - features are extracted automatically

### Train Model

```bash
cd scripts
python train.py
```

### Generate Charts

```bash
cd scripts
python generate_all_thesis_charts.py
```

## Requirements

- Python 3.11+
- TensorFlow 2.15+
- OpenCV
- Flask
- See requirements.txt for full list

## Data Format

- Videos: .mov, .mp4, .avi
- Features: Automatically extracted as .npy files
- Lists: Auto-generated 70/15/15 train/test/val split

## License

MIT License

## Author

SkateboardML Project Team
