# Git Branch Structure for SkateboardML Project

This document outlines the git branch organization for the SkateboardML project, with each branch corresponding to specific team member responsibilities and project tasks.

## Team Members and Responsibilities

### 1. **Trần Anh Tú (100%)**
- **Main Task**: Xây dựng mô hình CNN-LSTM, huấn luyện, viết chương 3
- **Branches**:
  - `tuta/cnn-lstm-model-development` - Main CNN-LSTM model architecture and development
  - `tuta/training-optimization` - Training scripts, optimization, and model improvements
  - `tuta/create-code-to-valid-model` - Model validation and testing code

### 2. **Nguyễn Thùy Trang (100%)**
- **Main Task**: Thu thập dữ liệu, tiền xử lý video, viết chương 2
- **Branches**:
  - `trang/data-collection-preprocessing` - Data collection, video preprocessing, feature extraction

### 3. **Nguyễn Tuấn Anh (100%)**
- **Main Task**: Đánh giá mô hình, viết chương 1, tổng hợp báo cáo
- **Branches**:
  - `anh/model-evaluation-chapter1` - Model evaluation, metrics, Chapter 1, final report

### 4. **Shared/Common**
- **Branches**:
  - `main` - Stable main branch with final integrated code
  - `web-application` - Web interface for demonstration and testing

## Branch Usage Guidelines

### Working with Branches

1. **Switch to your assigned branch**:
   ```bash
   git checkout [your-branch-name]
   ```

2. **Make your changes and commit**:
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

3. **Push your branch**:
   ```bash
   git push origin [your-branch-name]
   ```

4. **Merge to main when ready**:
   ```bash
   git checkout main
   git merge [your-branch-name]
   git push origin main
   ```

### Branch Descriptions

#### `tuta/cnn-lstm-model-development`
- **Purpose**: Core model architecture development
- **Contents**: 
  - CNN feature extraction models
  - LSTM sequence processing
  - Model architecture definitions
  - Core training loops

#### `tuta/training-optimization`
- **Purpose**: Training process optimization
- **Contents**:
  - Training scripts (train_windows.py, train_binary.py)
  - Hyperparameter tuning
  - Training optimization techniques
  - Model checkpointing and saving

#### `tuta/create-code-to-valid-model`
- **Purpose**: Model validation and testing
- **Contents**:
  - Validation scripts
  - Testing frameworks
  - Performance evaluation
  - Model verification tools

#### `trang/data-collection-preprocessing`
- **Purpose**: Data pipeline and preprocessing
- **Contents**:
  - Data collection scripts
  - Video preprocessing utilities
  - Feature extraction pipelines
  - Data organization tools

#### `anh/model-evaluation-chapter1`
- **Purpose**: Model evaluation and documentation
- **Contents**:
  - Performance metrics calculation
  - Model comparison scripts
  - Evaluation reports
  - Documentation and reports

#### `web-application`
- **Purpose**: Web interface for demonstration
- **Contents**:
  - Flask web application
  - HTML templates
  - User interface components
  - API endpoints

## File Organization by Branch

### Core Model Files (tuta branches)
```
- train_windows.py          # Main training script
- train_binary.py          # Binary classification training
- MLScript.py              # Alternative training script
- validation_tools.py      # Model validation utilities
```

### Data Processing Files (trang branch)
```
- generate_files.py        # Data generation utilities
- PopulateTrainTest.py    # Dataset splitting
- scripts/focus_binary.py  # Dataset focusing tools
- scripts/organize_data.py # Data organization
```

### Evaluation Files (anh branch)
```
- evaluation_metrics.py    # Performance metrics
- model_comparison.py     # Model comparison tools
- reports/                # Evaluation reports
- charts/                 # Performance visualizations
```

### Web Application Files (web-application branch)
```
- web_app.py              # Main Flask application
- templates/              # HTML templates
- static/                 # CSS/JS files
- requirements_web.txt    # Web dependencies
```

## Collaboration Workflow

1. **Individual Work**: Each team member works on their assigned branches
2. **Regular Commits**: Commit changes frequently with descriptive messages
3. **Branch Updates**: Regularly pull updates from main to keep branches current
4. **Code Reviews**: Review each other's code before merging to main
5. **Integration**: Merge completed features to main branch for integration testing

## Chapter Writing Guidelines

- **Chapter 1** (Anh): Introduction, literature review, problem statement
- **Chapter 2** (Trang): Data collection methodology, preprocessing techniques
- **Chapter 3** (Tú): Model architecture, training methodology, implementation

## Current Status

- ✅ **Project Setup**: Main structure and environment configured
- ✅ **Data Processing**: Binary classification dataset prepared (Ollie vs Kickflip)
- ✅ **Model Training**: CNN-LSTM model trained with optimization
- ✅ **Web Application**: Complete web interface implemented
- 🔄 **Model Validation**: In progress on `tuta/create-code-to-valid-model`
- 📝 **Documentation**: Ongoing across all branches

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tanhtuscode/dv-project.git
   cd dv-project
   ```

2. **Switch to your branch**:
   ```bash
   git checkout [your-branch-name]
   ```

3. **Set up Python environment**:
   ```bash
   python -m venv skateboard_env
   skateboard_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

4. **Start working on your assigned tasks**!

---

**Last Updated**: November 7, 2025
**Repository**: https://github.com/tanhtuscode/dv-project
**Main Branch**: `main`