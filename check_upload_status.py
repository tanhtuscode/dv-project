#!/usr/bin/env python3
"""Check status of uploaded videos and data lists"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config.paths import config

def check_upload_status():
    """Check if uploaded videos are in data lists"""
    tricks_dir = config.TRICKS_DIR
    
    print("="*60)
    print("CHECKING UPLOAD STATUS")
    print("="*60)
    
    # Get all .MOV and .npy files
    all_mov = []
    all_npy = []
    
    for label_dir in tricks_dir.iterdir():
        if label_dir.is_dir():
            mov_files = list(label_dir.glob("*.MOV")) + list(label_dir.glob("*.mov"))
            npy_files = list(label_dir.glob("*.npy"))
            
            all_mov.extend(mov_files)
            all_npy.extend(npy_files)
    
    print(f"\nTotal .MOV files: {len(all_mov)}")
    print(f"Total .npy files: {len(all_npy)}")
    
    # Check which .MOV files have corresponding .npy
    mov_with_npy = []
    mov_without_npy = []
    
    for mov_file in all_mov:
        npy_file = mov_file.with_suffix('.npy')
        if npy_file.exists():
            mov_with_npy.append(mov_file)
        else:
            mov_without_npy.append(mov_file)
    
    print(f"\n.MOV files WITH .npy: {len(mov_with_npy)}")
    print(f".MOV files WITHOUT .npy: {len(mov_without_npy)}")
    
    if mov_without_npy:
        print(f"\nFiles without features:")
        for f in mov_without_npy[:10]:
            print(f"  ❌ {f.relative_to(tricks_dir)}")
        if len(mov_without_npy) > 10:
            print(f"  ... and {len(mov_without_npy) - 10} more")
    
    # Read data lists
    train_list = []
    test_list = []
    val_list = []
    
    if config.TRAIN_LIST.exists():
        with open(config.TRAIN_LIST) as f:
            train_list = [line.strip() for line in f if line.strip()]
    
    if config.TEST_LIST.exists():
        with open(config.TEST_LIST) as f:
            test_list = [line.strip() for line in f if line.strip()]
    
    if config.VALIDATION_LIST.exists():
        with open(config.VALIDATION_LIST) as f:
            val_list = [line.strip() for line in f if line.strip()]
    
    all_in_lists = set(train_list + test_list + val_list)
    
    print(f"\n{'='*60}")
    print("DATA LISTS")
    print(f"{'='*60}")
    print(f"Train: {len(train_list)}")
    print(f"Test: {len(test_list)}")
    print(f"Validation: {len(val_list)}")
    print(f"Total in lists: {len(all_in_lists)}")
    
    # Check which processed files are NOT in lists
    missing_from_lists = []
    for mov_file in mov_with_npy:
        relative_path = str(mov_file.relative_to(tricks_dir)).replace('\\', '/')
        if relative_path not in all_in_lists:
            missing_from_lists.append(mov_file)
    
    print(f"\n{'='*60}")
    print("DISCREPANCIES")
    print(f"{'='*60}")
    
    if missing_from_lists:
        print(f"\n⚠️  {len(missing_from_lists)} processed files NOT in data lists:")
        for f in missing_from_lists[:10]:
            print(f"  ❗ {f.relative_to(tricks_dir)}")
        if len(missing_from_lists) > 10:
            print(f"  ... and {len(missing_from_lists) - 10} more")
        
        print(f"\n💡 Solution: Run update_data_lists() or restart web platform")
    else:
        print(f"\n✅ All processed videos are in data lists!")
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    check_upload_status()
