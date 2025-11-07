"""
SkateboardML Configuration
Dynamic path configuration that works on any PC
"""

import os
from pathlib import Path

class Config:
    """
    Centralized configuration for SkateboardML project
    Automatically detects project root and sets up paths
    """
    
    def __init__(self):
        # Find project root by looking for marker files
        self._project_root = self._find_project_root()
        self._setup_paths()
    
    def _find_project_root(self):
        """
        Find project root directory by looking for characteristic files
        """
        # Start from current file's directory
        current_dir = Path(__file__).parent
        
        # Look for project markers
        markers = [
            'BRANCH_STRUCTURE.md',
            '.gitignore',
            'data',
            'scripts',
            'app'
        ]
        
        # Search upwards for project root
        for path in [current_dir] + list(current_dir.parents):
            if all((path / marker).exists() for marker in markers[:3]):  # Check for key markers
                return path
        
        # Fallback: assume parent of scripts directory
        if current_dir.name == 'scripts':
            return current_dir.parent
        
        # Final fallback: current directory
        return current_dir
    
    def _setup_paths(self):
        """Setup all project paths"""
        # Core directories
        self.PROJECT_ROOT = self._project_root
        self.DATA_DIR = self.PROJECT_ROOT / 'data'
        self.SCRIPTS_DIR = self.PROJECT_ROOT / 'scripts'
        self.MODELS_DIR = self.PROJECT_ROOT / 'models'
        self.OUTPUTS_DIR = self.PROJECT_ROOT / 'outputs'
        self.APP_DIR = self.PROJECT_ROOT / 'app'
        self.CONFIG_DIR = self.PROJECT_ROOT / 'config'
        self.TOOLS_DIR = self.PROJECT_ROOT / 'tools'
        
        # Data paths
        self.TRICKS_DIR = self.DATA_DIR / 'Tricks'
        self.TRAIN_LIST = self.DATA_DIR / 'trainlist.txt'
        self.TEST_LIST = self.DATA_DIR / 'testlist.txt'
        self.VALIDATION_LIST = self.DATA_DIR / 'validationlist.txt'
        
        # App paths
        self.TEMPLATES_DIR = self.APP_DIR / 'templates'
        self.STATIC_DIR = self.APP_DIR / 'static'
        self.UPLOADS_DIR = self.APP_DIR / 'uploads'
        
        # Model paths
        self.BEST_MODEL = self.MODELS_DIR / 'best_model.keras'
        self.FINAL_MODEL = self.MODELS_DIR / 'final_model.keras'
        
        # Config paths
        self.REQUIREMENTS = self.CONFIG_DIR / 'requirements.txt'
        self.WEB_REQUIREMENTS = self.CONFIG_DIR / 'requirements_web.txt'
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs_to_create = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.OUTPUTS_DIR,
            self.UPLOADS_DIR
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def BASE_PATH(self):
        """For backward compatibility - returns Tricks directory"""
        return str(self.TRICKS_DIR)
    
    def get_path(self, path_name):
        """Get path by name - useful for dynamic access"""
        return getattr(self, path_name.upper(), None)
    
    def relative_to_project(self, path):
        """Convert absolute path to relative from project root"""
        try:
            return Path(path).relative_to(self.PROJECT_ROOT)
        except ValueError:
            return Path(path)
    
    def __str__(self):
        """String representation showing key paths"""
        return f"""SkateboardML Configuration:
Project Root: {self.PROJECT_ROOT}
Data Directory: {self.DATA_DIR}
Tricks Directory: {self.TRICKS_DIR}
Models Directory: {self.MODELS_DIR}
App Directory: {self.APP_DIR}
"""

# Global configuration instance
config = Config()

# Constants for easy access
SEQUENCE_LENGTH = 40

# Auto-detect labels with actual data (only folders with .npy files)
def get_labels_with_data():
    """Dynamically get labels that have actual data files"""
    tricks_dir = config.TRICKS_DIR
    labels_with_data = []
    
    if tricks_dir.exists():
        for folder in sorted(tricks_dir.iterdir()):
            if folder.is_dir():
                # Check if folder has any .npy files
                npy_files = list(folder.glob('*.npy'))
                if npy_files:
                    labels_with_data.append(folder.name)
    
    # Fallback to default if no data found
    if not labels_with_data:
        labels_with_data = ["Kickflip", "Ollie"]
    
    return labels_with_data

LABELS = get_labels_with_data()
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# Path shortcuts for common use
PROJECT_ROOT = config.PROJECT_ROOT
DATA_DIR = config.DATA_DIR
TRICKS_DIR = config.TRICKS_DIR
MODELS_DIR = config.MODELS_DIR
OUTPUTS_DIR = config.OUTPUTS_DIR
APP_DIR = config.APP_DIR

# For backward compatibility
BASE_PATH = config.BASE_PATH

if __name__ == "__main__":
    print(config)
    print(f"\nAll paths exist: {all(path.exists() for path in [DATA_DIR, TRICKS_DIR, MODELS_DIR])}")