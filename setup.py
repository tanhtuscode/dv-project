"""
SkateboardML - Automated Setup Script
Automatically sets up the complete environment, dependencies, and directory structure
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class SkateboardMLSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_executable = sys.executable
        self.os_type = platform.system()
        
    def print_header(self, message):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {message}")
        print("="*70 + "\n")
    
    def print_step(self, step_num, message):
        """Print step information"""
        print(f"\n[{step_num}] {message}")
        print("-" * 70)
    
    def run_command(self, command, description):
        """Run a shell command and report status"""
        print(f"  → {description}...")
        try:
            if isinstance(command, list):
                result = subprocess.run(command, check=True, capture_output=True, text=True)
            else:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"  ✓ Success!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e}")
            if e.stderr:
                print(f"    Error: {e.stderr[:200]}")
            return False
    
    def create_directories(self):
        """Create all required directories"""
        self.print_step(1, "Creating Directory Structure")
        
        directories = [
            'data/Tricks/Kickflip',
            'data/Tricks/Ollie',
            'data/Tricks/Shuvit',
            'data/Tricks/Varial',
            'data/Tricks/Front180',
            'data/Tricks/Back180',
            'data/Tricks/Frontshuvit',
            'models',
            'outputs/thesis_charts',
            'outputs/advanced_charts',
            'app/uploads',
            'app/static',
            'app/templates',
            'logs'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {directory}")
        
        print(f"\n  Total: {len(directories)} directories created/verified")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_step(2, "Installing Python Dependencies")
        
        # Check if requirements.txt exists
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            print("  ⚠ requirements.txt not found. Creating one...")
            self.create_requirements_file()
        
        # Upgrade pip
        self.run_command(
            [self.python_executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            "Upgrading pip"
        )
        
        # Install requirements
        self.run_command(
            [self.python_executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
            "Installing dependencies from requirements.txt"
        )
    
    def create_requirements_file(self):
        """Create requirements.txt if it doesn't exist"""
        requirements = """# Core ML/DL Libraries
tensorflow>=2.15.0
keras>=2.15.0
numpy>=1.24.0
opencv-python>=4.8.0
scikit-learn>=1.3.0

# Web Framework
flask>=3.0.0
werkzeug>=3.0.0

# Data Processing
pandas>=2.1.0
pillow>=10.0.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Utilities
tqdm>=4.66.0
"""
        
        requirements_file = self.project_root / 'requirements.txt'
        with open(requirements_file, 'w') as f:
            f.write(requirements)
        print(f"  ✓ Created requirements.txt")
    
    def create_config_files(self):
        """Create configuration files"""
        self.print_step(3, "Creating Configuration Files")
        
        # Create .gitignore if it doesn't exist
        gitignore_file = self.project_root / '.gitignore'
        if not gitignore_file.exists():
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.mov
*.MOV
*.npy
models/*.keras
models/*.h5
logs/*.log
app/uploads/*
!app/uploads/.gitkeep

# Data lists (generated)
data/testlist.txt
data/trainlist.txt
data/validationlist.txt
"""
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_content)
            print(f"  ✓ Created .gitignore")
        else:
            print(f"  ✓ .gitignore already exists")
        
        # Create .gitkeep files for empty directories
        keepfiles = [
            'app/uploads/.gitkeep',
            'app/static/.gitkeep',
            'logs/.gitkeep'
        ]
        
        for keepfile in keepfiles:
            keepfile_path = self.project_root / keepfile
            keepfile_path.touch(exist_ok=True)
            print(f"  ✓ Created {keepfile}")
    
    def initialize_data_lists(self):
        """Create initial empty data list files"""
        self.print_step(4, "Initializing Data List Files")
        
        data_dir = self.project_root / 'data'
        
        list_files = ['trainlist.txt', 'testlist.txt', 'validationlist.txt']
        
        for list_file in list_files:
            list_path = data_dir / list_file
            if not list_path.exists():
                list_path.touch()
                print(f"  ✓ Created {list_file}")
            else:
                print(f"  ✓ {list_file} already exists")
    
    def create_startup_scripts(self):
        """Create startup scripts for different platforms"""
        self.print_step(5, "Creating Startup Scripts")
        
        # Windows batch file
        if self.os_type == 'Windows':
            batch_content = f"""@echo off
echo Starting SkateboardML Web Platform...
cd /d "{self.project_root}\\app"
"{self.python_executable}" web_platform.py
pause
"""
            batch_file = self.project_root / 'start_web_app.bat'
            with open(batch_file, 'w') as f:
                f.write(batch_content)
            print(f"  ✓ Created start_web_app.bat")
        
        # Unix shell script
        sh_content = f"""#!/bin/bash
echo "Starting SkateboardML Web Platform..."
cd "{self.project_root}/app"
"{self.python_executable}" web_platform.py
"""
        sh_file = self.project_root / 'start_web_app.sh'
        with open(sh_file, 'w') as f:
            f.write(sh_content)
        
        # Make shell script executable on Unix
        if self.os_type in ['Linux', 'Darwin']:
            os.chmod(sh_file, 0o755)
        print(f"  ✓ Created start_web_app.sh")
    
    def create_readme(self):
        """Create or update README with setup instructions"""
        self.print_step(6, "Creating README Documentation")
        
        readme_content = """# SkateboardML - Skateboard Trick Classification

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
"""
        
        readme_file = self.project_root / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"  ✓ Created/Updated README.md")
    
    def verify_installation(self):
        """Verify that everything is set up correctly"""
        self.print_step(7, "Verifying Installation")
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 11:
            print(f"  ✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            checks.append(True)
        else:
            print(f"  ✗ Python version too old: {python_version.major}.{python_version.minor}")
            checks.append(False)
        
        # Check critical packages
        critical_packages = ['tensorflow', 'flask', 'cv2', 'numpy']
        for package in critical_packages:
            try:
                __import__(package)
                print(f"  ✓ {package} installed")
                checks.append(True)
            except ImportError:
                print(f"  ✗ {package} not found")
                checks.append(False)
        
        # Check directory structure
        critical_dirs = ['app', 'data', 'models', 'scripts', 'config']
        for directory in critical_dirs:
            if (self.project_root / directory).exists():
                print(f"  ✓ Directory '{directory}' exists")
                checks.append(True)
            else:
                print(f"  ✗ Directory '{directory}' missing")
                checks.append(False)
        
        # Check critical files
        critical_files = ['app/web_platform.py', 'config/paths.py']
        for file in critical_files:
            if (self.project_root / file).exists():
                print(f"  ✓ File '{file}' exists")
                checks.append(True)
            else:
                print(f"  ✗ File '{file}' missing")
                checks.append(False)
        
        return all(checks)
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_header("SkateboardML - Automated Setup")
        
        print(f"Project Root: {self.project_root}")
        print(f"Python: {self.python_executable}")
        print(f"OS: {self.os_type}")
        
        try:
            self.create_directories()
            self.install_dependencies()
            self.create_config_files()
            self.initialize_data_lists()
            self.create_startup_scripts()
            self.create_readme()
            
            # Verify installation
            if self.verify_installation():
                self.print_header("✓ Setup Completed Successfully!")
                print("\nNext Steps:")
                print("1. Add your video files to data/Tricks/<LabelName>/")
                print("2. Run the web platform:")
                if self.os_type == 'Windows':
                    print("   → Double-click start_web_app.bat")
                else:
                    print("   → ./start_web_app.sh")
                print("3. Open http://localhost:5000 in your browser")
                print("\nHappy coding! 🛹\n")
            else:
                self.print_header("⚠ Setup Completed with Warnings")
                print("\nSome components may not be properly installed.")
                print("Please check the messages above and install missing dependencies.\n")
        
        except Exception as e:
            self.print_header("✗ Setup Failed")
            print(f"\nError: {e}")
            print("\nPlease check the error message and try again.\n")
            sys.exit(1)

def main():
    """Main entry point"""
    setup = SkateboardMLSetup()
    setup.run_setup()

if __name__ == '__main__':
    main()
