"""
Organize SkateboardML data into proper train/validation/test structure.
This script reorganizes the data based on existing train/test lists and creates a validation split.
"""

import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Base paths
BASE_PATH = Path("d:/DV/SkateboardML")
TRICKS_PATH = BASE_PATH / "Tricks"
DATA_PATH = BASE_PATH / "data"

# Create data directories
for split in ["train", "validation", "test"]:
    (DATA_PATH / split).mkdir(parents=True, exist_ok=True)

def load_file_list(filename):
    """Load file list from train/test list files."""
    filepath = BASE_PATH / filename
    if not filepath.exists():
        print(f"Warning: {filename} not found")
        return []
    
    with open(filepath, 'r') as f:
        file_list = [row.strip() for row in f if row.strip()]
        # Remove class labels if present (format: "path class_id")
        file_list = [row.split(' ')[0] for row in file_list]
    return file_list

def get_label_from_path(filepath):
    """Extract label from file path."""
    return Path(filepath).parent.name

def organize_files():
    """Organize files into train/validation/test splits."""
    
    # Load existing splits
    train_files = load_file_list('trainlist03.txt')
    test_files = load_file_list('testlist03.txt')
    
    print(f"Found {len(train_files)} training files")
    print(f"Found {len(test_files)} test files")
    
    # Create validation split from training data (20% of training data)
    random.shuffle(train_files)
    val_split = int(len(train_files) * 0.2)
    validation_files = train_files[:val_split]
    train_files = train_files[val_split:]
    
    print(f"Split into:")
    print(f"  Training: {len(train_files)} files")
    print(f"  Validation: {len(validation_files)} files") 
    print(f"  Test: {len(test_files)} files")
    
    # Process each split
    splits = {
        'train': train_files,
        'validation': validation_files,
        'test': test_files
    }
    
    for split_name, file_list in splits.items():
        print(f"\nProcessing {split_name} split...")
        split_path = DATA_PATH / split_name
        
        # Group files by class
        class_counts = {}
        
        for filepath in file_list:
            label = get_label_from_path(filepath)
            
            # Create class directory
            class_dir = split_path / label
            class_dir.mkdir(exist_ok=True)
            
            # Copy .mov file if it exists
            source_mov = TRICKS_PATH / filepath
            if source_mov.exists():
                dest_mov = class_dir / source_mov.name
                if not dest_mov.exists():
                    shutil.copy2(source_mov, dest_mov)
                    print(f"  Copied: {source_mov.name} -> {split_name}/{label}/")
            else:
                print(f"  Warning: Missing video file: {source_mov}")
            
            # Copy .npy file if it exists
            npy_file = source_mov.with_suffix('.npy')
            if npy_file.exists():
                dest_npy = class_dir / npy_file.name
                if not dest_npy.exists():
                    shutil.copy2(npy_file, dest_npy)
            else:
                print(f"  Warning: Missing feature file: {npy_file}")
            
            # Count files per class
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Print class distribution
        print(f"  Class distribution in {split_name}:")
        for label, count in sorted(class_counts.items()):
            print(f"    {label}: {count}")

def create_summary_files():
    """Create summary files for each split."""
    for split in ["train", "validation", "test"]:
        split_path = DATA_PATH / split
        summary_file = split_path / f"{split}_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"# {split.upper()} SET SUMMARY\n")
            f.write(f"Generated on: {__import__('datetime').datetime.now()}\n\n")
            
            total_files = 0
            for class_dir in split_path.iterdir():
                if class_dir.is_dir() and class_dir.name != '__pycache__':
                    mov_files = list(class_dir.glob("*.mov"))
                    npy_files = list(class_dir.glob("*.npy"))
                    
                    f.write(f"{class_dir.name}:\n")
                    f.write(f"  Video files (.mov): {len(mov_files)}\n")
                    f.write(f"  Feature files (.npy): {len(npy_files)}\n")
                    f.write(f"  Total: {len(mov_files)}\n\n")
                    
                    total_files += len(mov_files)
            
            f.write(f"TOTAL FILES: {total_files}\n")

def clean_old_structure():
    """Clean up old file structure."""
    # Move original lists to archive
    archive_dir = BASE_PATH / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    for filename in ["trainlist02.txt", "trainlist03.txt", "testlist02.txt", "testlist03.txt"]:
        src = BASE_PATH / filename
        if src.exists():
            shutil.move(str(src), str(archive_dir / filename))
            print(f"Moved {filename} to archive/")

if __name__ == "__main__":
    print("=" * 60)
    print("SkateboardML Data Organization Script")
    print("=" * 60)
    
    # Organize files
    organize_files()
    
    # Create summary files
    print("\nCreating summary files...")
    create_summary_files()
    
    # Clean old structure
    print("\nCleaning up old structure...")
    clean_old_structure()
    
    print("\n" + "=" * 60)
    print("Data organization complete!")
    print("=" * 60)
    print("\nNew structure:")
    print("  data/")
    print("    train/")
    print("    validation/")
    print("    test/")
    print("  models/ (for saved models)")
    print("  outputs/ (for results, charts, logs)")
    print("  archive/ (old train/test lists)")