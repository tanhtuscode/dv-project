"""
Filter SkateboardML data to focus only on Ollie and Kickflip tricks.
This script creates new train/test lists with only Ollie and Kickflip samples.
"""

import os
from pathlib import Path

# Base paths
BASE_PATH = Path("d:/DV/SkateboardML")
TRICKS_PATH = BASE_PATH / "Tricks"

# Target classes
TARGET_CLASSES = ["Ollie", "Kickflip"]

def filter_file_list(input_filename, output_filename):
    """Filter file list to only include target classes."""
    input_path = BASE_PATH / input_filename
    output_path = BASE_PATH / output_filename
    
    if not input_path.exists():
        print(f"Warning: {input_filename} not found")
        return 0
    
    filtered_files = []
    class_counts = {cls: 0 for cls in TARGET_CLASSES}
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Extract file path and class
            parts = line.split()
            filepath = parts[0]
            
            # Get class from file path
            class_name = Path(filepath).parent.name
            
            # Only keep Ollie and Kickflip
            if class_name in TARGET_CLASSES:
                filtered_files.append(line)
                class_counts[class_name] += 1
    
    # Write filtered list
    with open(output_path, 'w') as f:
        for line in filtered_files:
            f.write(line + '\n')
    
    print(f"Filtered {input_filename} -> {output_filename}:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    return sum(class_counts.values())

def create_binary_lists():
    """Create binary classification train/test lists."""
    print("=" * 50)
    print("Creating Binary Classification Lists (Ollie vs Kickflip)")
    print("=" * 50)
    
    # Filter existing lists
    train_count = filter_file_list("trainlist03.txt", "trainlist_binary.txt")
    test_count = filter_file_list("testlist03.txt", "testlist_binary.txt")
    
    print(f"\nTotal filtered files:")
    print(f"  Training: {train_count}")
    print(f"  Test: {test_count}")
    
    return train_count, test_count

def verify_files_exist():
    """Verify that the video and feature files actually exist."""
    print("\n" + "=" * 50)
    print("Verifying File Existence")
    print("=" * 50)
    
    for class_name in TARGET_CLASSES:
        class_dir = TRICKS_PATH / class_name
        if not class_dir.exists():
            print(f"Warning: {class_name} directory not found")
            continue
            
        mov_files = list(class_dir.glob("*.mov"))
        npy_files = list(class_dir.glob("*.npy"))
        
        print(f"\n{class_name}:")
        print(f"  Video files (.mov): {len(mov_files)}")
        print(f"  Feature files (.npy): {len(npy_files)}")
        
        # Check for missing .npy files
        missing_npy = []
        for mov_file in mov_files:
            npy_file = mov_file.with_suffix('.npy')
            if not npy_file.exists():
                missing_npy.append(mov_file.name)
        
        if missing_npy:
            print(f"  Missing .npy files: {len(missing_npy)}")
            if len(missing_npy) <= 10:  # Show first 10
                for name in missing_npy:
                    print(f"    - {name}")
            else:
                print(f"    (showing first 10)")
                for name in missing_npy[:10]:
                    print(f"    - {name}")

def update_training_scripts():
    """Update the LABELS in training scripts to only include Ollie and Kickflip."""
    scripts_to_update = [
        "train_windows.py",
        "train_binary.py",
        "MLScript.py"
    ]
    
    print("\n" + "=" * 50)
    print("Updating Training Scripts")
    print("=" * 50)
    
    for script_name in scripts_to_update:
        script_path = BASE_PATH / script_name
        if not script_path.exists():
            print(f"  {script_name}: Not found")
            continue
            
        # Read the file
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Update LABELS line
        old_labels = 'LABELS = ["Back180", "Front180", "Frontshuvit", "Kickflip", "Ollie", "Shuvit", "Varial"]'
        new_labels = 'LABELS = ["Kickflip", "Ollie"]'
        
        if old_labels in content:
            content = content.replace(old_labels, new_labels)
            
            # Write back
            with open(script_path, 'w') as f:
                f.write(content)
            
            print(f"  {script_name}: Updated LABELS to binary classification")
        else:
            print(f"  {script_name}: LABELS not found or already updated")

def create_project_summary():
    """Create a summary of the focused project."""
    summary_path = BASE_PATH / "PROJECT_SUMMARY.md"
    
    with open(summary_path, 'w') as f:
        f.write("# SkateboardML - Binary Classification Project\n\n")
        f.write("## Project Focus\n")
        f.write("This project has been focused on binary classification of skateboarding tricks:\n")
        f.write("- **Ollie**: Basic skateboard jump\n")
        f.write("- **Kickflip**: Ollie with board flip\n\n")
        
        f.write("## Data Structure\n")
        f.write("```\n")
        f.write("Tricks/\n")
        f.write("  Kickflip/          # Kickflip video and feature files\n")
        f.write("  Ollie/             # Ollie video and feature files\n\n")
        
        f.write("trainlist_binary.txt  # Training file list (Ollie + Kickflip only)\n")
        f.write("testlist_binary.txt   # Test file list (Ollie + Kickflip only)\n")
        f.write("```\n\n")
        
        f.write("## Updated Scripts\n")
        f.write("- `train_binary.py`: Binary classification training\n")
        f.write("- `train_windows.py`: Updated for binary classification\n")
        f.write("- `MLScript.py`: Updated for binary classification\n\n")
        
        f.write("## Next Steps\n")
        f.write("1. Run `python train_binary.py` for training\n")
        f.write("2. Use `scripts/generate_charts.py` for visualization\n")
        f.write("3. Deploy with `app.py` Flask application\n")
    
    print(f"  Created: PROJECT_SUMMARY.md")

if __name__ == "__main__":
    print("SkateboardML - Focusing on Binary Classification")
    print("Target classes: Ollie and Kickflip")
    
    # Create binary file lists
    train_count, test_count = create_binary_lists()
    
    # Verify files exist
    verify_files_exist()
    
    # Update training scripts
    update_training_scripts()
    
    # Create project summary
    create_project_summary()
    
    print("\n" + "=" * 50)
    print("Project Focus Complete!")
    print("=" * 50)
    print(f"✅ Removed: Back180, Front180, Frontshuvit, Shuvit, Varial")
    print(f"✅ Kept: Ollie and Kickflip ({train_count + test_count} total samples)")
    print(f"✅ Created: trainlist_binary.txt, testlist_binary.txt")
    print(f"✅ Updated: Training scripts for binary classification")
    print(f"✅ Created: PROJECT_SUMMARY.md")