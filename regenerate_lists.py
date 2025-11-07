#!/usr/bin/env python3
"""Regenerate data lists using web platform's update function"""

import sys
from pathlib import Path

# Don't load TensorFlow stuff
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(str(Path(__file__).parent))

# Import only what we need
from config.paths import config
import numpy as np

def update_data_lists_standalone():
    """Update data lists without loading TensorFlow"""
    tricks_dir = config.TRICKS_DIR
    all_files = []
    
    print("Scanning for videos with features...")
    
    # Only process folders that have .npy files
    for label_dir in tricks_dir.iterdir():
        if label_dir.is_dir():
            npy_files = list(label_dir.glob("*.npy"))
            if npy_files:
                print(f"  {label_dir.name}: {len(npy_files)} features")
                for npy_file in npy_files:
                    # Look for corresponding .MOV or .mov file
                    mov_file_upper = npy_file.with_suffix('.MOV')
                    mov_file_lower = npy_file.with_suffix('.mov')
                    
                    mov_file = None
                    if mov_file_upper.exists():
                        mov_file = mov_file_upper
                    elif mov_file_lower.exists():
                        mov_file = mov_file_lower
                    
                    if mov_file:
                        relative_path = str(mov_file.relative_to(tricks_dir)).replace('\\', '/')
                        all_files.append(relative_path)
    
    print(f"\nTotal videos with features: {len(all_files)}")
    
    # Split data: 70% train, 15% test, 15% validation
    np.random.shuffle(all_files)
    total = len(all_files)
    train_size = int(0.7 * total)
    test_size = int(0.15 * total)
    
    train_files = all_files[:train_size]
    test_files = all_files[train_size:train_size + test_size]
    val_files = all_files[train_size + test_size:]
    
    # Write files
    print(f"\nWriting data lists...")
    print(f"  Train: {len(train_files)}")
    print(f"  Test: {len(test_files)}")
    print(f"  Validation: {len(val_files)}")
    
    with open(config.TRAIN_LIST, 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")
    
    with open(config.TEST_LIST, 'w') as f:
        for file in test_files:
            f.write(f"{file}\n")
    
    with open(config.VALIDATION_LIST, 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")
    
    print(f"\n✅ Data lists updated successfully!")
    print(f"   {config.TRAIN_LIST}")
    print(f"   {config.TEST_LIST}")
    print(f"   {config.VALIDATION_LIST}")

if __name__ == '__main__':
    update_data_lists_standalone()
