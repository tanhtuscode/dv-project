# SkateboardML Project Structure

This document outlines the organized file structure of the SkateboardML project after cleanup and reorganization.

## 📁 **Root Directory Structure**

```
SkateboardML/
├── 📄 .gitignore                    # Git ignore rules
├── 📄 BRANCH_STRUCTURE.md           # Git branch organization
├── 📄 PROJECT_STRUCTURE.md          # This file - project organization
├── 📁 app/                          # Web Application
├── 📁 config/                       # Configuration Files
├── 📁 data/                         # All Data Files
├── 📁 explorations/                 # Research & Experiments
├── 📁 models/                       # Trained Models (gitignored)
├── 📁 outputs/                      # Generated Outputs (gitignored)
├── 📁 scripts/                      # All Python Scripts
└── 📁 tools/                        # Utility Tools & Startup Scripts
```

## 📂 **Detailed Folder Contents**

### 🌐 **app/** - Web Application
```
app/
├── web_app.py                      # Main Flask web application
├── static/                         # CSS, JS, images
├── templates/                      # HTML templates
│   ├── index.html                  # Dashboard
│   ├── data_collection.html        # Data upload interface
│   ├── model_testing.html          # Model testing interface
│   └── prediction.html             # Prediction interface
└── uploads/                        # Temporary upload storage
```

**Purpose**: Complete web interface for data collection, model testing, and prediction.

### ⚙️ **config/** - Configuration Files
```
config/
├── requirements.txt                # Core Python dependencies
└── requirements_web.txt            # Web application dependencies
```

**Purpose**: All configuration and dependency management files.

### 💾 **data/** - All Data Files
```
data/
├── trainlist_binary.txt           # Training file list (Ollie vs Kickflip)
├── testlist_binary.txt            # Test file list (Ollie vs Kickflip)
├── Tricks/                        # Video data and features
│   ├── Kickflip/                  # Kickflip videos and .npy features
│   │   ├── Kickflip0.mov
│   │   ├── Kickflip0.npy
│   │   └── ... (114 videos total)
│   └── Ollie/                     # Ollie videos and .npy features
│       ├── Ollie0.mov
│       ├── Ollie0.npy
│       └── ... (108 videos total)
├── train/                         # Future: organized training data
├── test/                          # Future: organized test data
└── validation/                    # Future: organized validation data
```

**Purpose**: All datasets, file lists, and video content organized by trick type.

### 🔬 **explorations/** - Research & Experiments
```
explorations/
├── cppn/                          # CPPN experiments
│   └── sample.png
└── mnist/                         # MNIST experiments (Julia)
    ├── conv.jl
    ├── Manifest.toml
    └── Project.toml
```

**Purpose**: Research experiments and alternative approaches.

### 🤖 **models/** - Trained Models
```
models/
├── (best_model.keras)             # Best performing model (gitignored)
├── (final_model.keras)            # Final trained model (gitignored)
└── (model_checkpoints/)           # Training checkpoints (gitignored)
```

**Purpose**: Stores trained models (not in git due to size).

### 📊 **outputs/** - Generated Outputs
```
outputs/
├── (charts/)                      # Performance charts (gitignored)
├── (logs/)                        # Training logs (gitignored)
└── (reports/)                     # Evaluation reports (gitignored)
```

**Purpose**: Generated outputs, charts, and reports (not in git).

### 📝 **scripts/** - All Python Scripts
```
scripts/
├── train_windows.py               # Main training script
├── train_binary.py                # Binary classification training
├── count_labels.py                # Dataset analysis
├── evaluate_model.py              # Model evaluation
├── focus_binary.py                # Dataset filtering for binary classification
├── generate_charts.py             # Performance visualization
└── organize_data.py               # Data organization utilities
```

**Purpose**: All Python scripts for training, evaluation, and data processing.

### 🔧 **tools/** - Utility Tools
```
tools/
└── start_web_app.bat              # Windows startup script for web app
```

**Purpose**: Utility scripts and tools for project management.

## 🎯 **Usage by Team Member**

### **Trần Anh Tú** - Model Development (Chapter 3)
**Primary Directories**: 
- `scripts/` - Training and model development scripts
- `models/` - Saved trained models
- `outputs/` - Training logs and performance metrics

**Key Files**:
- `scripts/train_windows.py` - Main training script
- `scripts/train_binary.py` - Binary classification
- `scripts/evaluate_model.py` - Model evaluation

### **Nguyễn Thùy Trang** - Data Collection (Chapter 2)
**Primary Directories**:
- `data/` - All datasets and file organization
- `scripts/organize_data.py` - Data processing scripts
- `app/` - Web interface for data collection

**Key Files**:
- `data/Tricks/` - Video datasets
- `data/trainlist_binary.txt` - Training file list
- `app/templates/data_collection.html` - Data upload interface

### **Nguyễn Tuấn Anh** - Model Evaluation (Chapter 1)
**Primary Directories**:
- `scripts/` - Evaluation and analysis scripts
- `outputs/` - Generated reports and charts
- `app/` - Web interface for testing

**Key Files**:
- `scripts/evaluate_model.py` - Model evaluation
- `scripts/generate_charts.py` - Performance visualization
- `app/templates/model_testing.html` - Testing interface

## 🚀 **Getting Started**

### **1. Environment Setup**
```bash
# Install core dependencies
pip install -r config/requirements.txt

# Install web dependencies (if using web app)
pip install -r config/requirements_web.txt
```

### **2. Training Models**
```bash
cd scripts
python train_windows.py      # Full training
python train_binary.py       # Binary classification only
```

### **3. Running Web Application**
```bash
# Option 1: Use startup script
tools/start_web_app.bat

# Option 2: Manual start
cd app
python web_app.py
```

### **4. Model Evaluation**
```bash
cd scripts
python evaluate_model.py     # Generate evaluation metrics
python generate_charts.py    # Create performance charts
```

## 📋 **File Path Updates**

Due to reorganization, the following paths have been updated:

| Old Path | New Path | Purpose |
|----------|----------|---------|
| `train_windows.py` | `scripts/train_windows.py` | Training script |
| `train_binary.py` | `scripts/train_binary.py` | Binary training |
| `web_app.py` | `app/web_app.py` | Web application |
| `templates/` | `app/templates/` | HTML templates |
| `Tricks/` | `data/Tricks/` | Video data |
| `trainlist_binary.txt` | `data/trainlist_binary.txt` | Training list |
| `testlist_binary.txt` | `data/testlist_binary.txt` | Test list |
| `requirements.txt` | `config/requirements.txt` | Dependencies |

## 🔧 **Configuration Updates**

All scripts have been updated to use the new paths:
- Training scripts now look for data in `../data/`
- Web application updated to use `data/Tricks/` and `data/*.txt`
- Startup scripts updated to run from correct directories

## 📝 **Notes**

- **Models and outputs** are gitignored due to size
- **Data organization** follows standard ML project structure
- **Scripts** are centralized for easy access and execution
- **Web application** is self-contained in `app/` folder
- **Configuration** is centralized in `config/` folder

---

**Last Updated**: November 7, 2025  
**Project**: SkateboardML Binary Classification (Ollie vs Kickflip)  
**Status**: Organized and Ready for Development