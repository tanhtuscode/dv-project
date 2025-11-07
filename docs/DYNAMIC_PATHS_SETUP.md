# Dynamic Path Configuration System

## Overview
The SkateboardML project now uses a centralized, dynamic path configuration system that automatically detects the project root directory and sets up all paths relative to it. This allows the project to work on any PC without manual path modifications.

## How It Works

### Automatic Project Detection
The `config/paths.py` module automatically finds the project root by looking for these marker files:
- `BRANCH_STRUCTURE.md`
- `.gitignore`
- `data/` folder

It searches up from the script location until it finds these markers, ensuring it works regardless of where the script is called from.

### Configuration Class
The `ProjectPaths` class in `config/paths.py` provides:
- **Dynamic path resolution**: All paths calculated relative to detected project root
- **Cross-platform compatibility**: Uses `pathlib.Path` for Windows/Linux/Mac compatibility
- **Automatic directory creation**: Creates required directories if they don't exist
- **Centralized constants**: Single source of truth for SEQUENCE_LENGTH, LABELS, etc.

### Updated Files

#### Core Configuration
- **config/paths.py**: Central configuration module with auto-detection

#### Training Scripts
- **scripts/train_windows.py**: Updated to use `from config.paths import config, SEQUENCE_LENGTH, LABELS`
- **scripts/train_binary.py**: Updated with dynamic path imports
- **train_windows.py** (root): Legacy file updated for compatibility

#### Utility Scripts
- **scripts/count_labels.py**: Updated to use `config.TRAIN_LIST` and `config.TEST_LIST`
- **scripts/evaluate_model.py**: Updated to use `config.BEST_MODEL` and centralized LABELS
- **scripts/generate_charts.py**: Updated to use `config.OUTPUTS_DIR` and dynamic paths
- **scripts/focus_binary.py**: Updated to use `config.PROJECT_ROOT` and `config.TRICKS_DIR`
- **scripts/organize_data.py**: Updated to use `config.DATA_DIR` and `config.TRICKS_DIR`

#### Web Application
- **app/web_app.py**: Updated to use centralized configuration for all paths

## Usage

### In Your Scripts
Instead of hard-coded paths:
```python
# OLD - Hard-coded paths
BASE_PATH = 'd:/DV/SkateboardML'
TRICKS_PATH = os.path.join(BASE_PATH, 'Tricks')
```

Use dynamic imports:
```python
# NEW - Dynamic paths
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import config, SEQUENCE_LENGTH, LABELS

# Use configuration paths
TRICKS_PATH = config.TRICKS_DIR
DATA_PATH = config.DATA_DIR
```

### Available Paths
The `config` object provides these pre-configured paths:

**Project Structure:**
- `config.PROJECT_ROOT` - Main project directory
- `config.BASE_PATH` - Backward compatibility alias

**Data Paths:**
- `config.DATA_DIR` - data/ directory
- `config.TRICKS_DIR` - data/Tricks/ directory
- `config.TRAIN_LIST` - data/trainlist_binary.txt
- `config.TEST_LIST` - data/testlist_binary.txt

**Application Paths:**
- `config.APP_DIR` - app/ directory
- `config.TEMPLATES_DIR` - app/templates/
- `config.STATIC_DIR` - app/static/
- `config.UPLOADS_DIR` - app/uploads/

**Model Paths:**
- `config.MODELS_DIR` - models/ directory
- `config.BEST_MODEL` - models/best_model.keras
- `config.FINAL_MODEL` - models/final_model.keras

**Output Paths:**
- `config.OUTPUTS_DIR` - outputs/ directory
- `config.CHARTS_DIR` - outputs/charts/
- `config.LOGS_DIR` - outputs/logs/

**Script Paths:**
- `config.SCRIPTS_DIR` - scripts/ directory

**Tool Paths:**
- `config.TOOLS_DIR` - tools/ directory

**Config Paths:**
- `config.CONFIG_DIR` - config/ directory

### Available Constants
- `SEQUENCE_LENGTH` - Video sequence length (40 frames)
- `LABELS` - List of trick labels (["Kickflip", "Ollie"])
- `ALLOWED_EXTENSIONS` - Set of allowed video file extensions

## Benefits

1. **Portability**: Works on any PC without modifying code
2. **Maintainability**: Single source of truth for all paths
3. **Consistency**: All scripts use the same path structure
4. **Safety**: Automatic directory creation prevents missing folder errors
5. **Flexibility**: Easy to reorganize project structure by updating one file

## Running on Different PCs

### Setup on New PC
1. Clone the repository: `git clone <repo-url>`
2. Navigate to project: `cd SkateboardML`
3. Install dependencies: `pip install -r requirements.txt`
4. Run any script - paths will auto-detect!

### No Manual Configuration Required
The system automatically:
- Detects the project root directory
- Sets up all paths relative to that root
- Creates missing directories
- Works on Windows, Linux, and Mac

## Example: Running Training

On **any PC**, from **any location** within the project:

```powershell
# From project root
python scripts/train_binary.py

# From scripts directory
cd scripts
python train_binary.py

# From any subdirectory
cd explorations/mnist
python ../../scripts/train_binary.py
```

All commands work identically - the configuration system handles path resolution!

## Testing the Configuration

Run the configuration module directly to verify paths:
```powershell
python config/paths.py
```

This will print:
- Detected project root
- All configured paths
- Verification that directories exist

## Troubleshooting

### If Auto-Detection Fails
The system looks for these files to identify the project root:
- `BRANCH_STRUCTURE.md`
- `.gitignore`
- `data/` directory

Ensure at least one of these exists in your project root.

### Manual Override (Not Recommended)
If needed, you can manually set the project root in `config/paths.py`:
```python
# In config/paths.py __init__ method
self.PROJECT_ROOT = Path("C:/path/to/your/project")
```

But the auto-detection should work in 99% of cases!

## Migration Complete ✅

All scripts and applications in SkateboardML now use the dynamic path configuration system. The project is fully portable and ready to use on multiple development machines without any code modifications.
