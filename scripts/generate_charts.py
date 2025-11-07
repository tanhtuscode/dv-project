"""
Generate visualization charts for the SkateboardML project.
Creates charts for class distribution, training history, and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Base paths
BASE_PATH = "d:/DV/SkateboardML"
OUTPUT_DIR = os.path.join(BASE_PATH, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["Back180", "Front180", "Frontshuvit", "Kickflip", "Ollie", "Shuvit", "Varial"]


def load_file_list(filename):
    """Load train or test file list."""
    filepath = os.path.join(BASE_PATH, filename)
    with open(filepath, 'r') as f:
        file_list = [row.strip() for row in f]
        file_list = [row.split(' ')[0] for row in file_list]
    return file_list


def extract_labels(file_list):
    """Extract labels from file paths."""
    labels = []
    for path in file_list:
        label = os.path.basename(os.path.dirname(path))
        labels.append(label)
    return labels


def plot_class_distribution():
    """Plot class distribution for train and test sets."""
    print("Generating class distribution chart...")
    
    # Load data
    train_list = load_file_list('trainlist03.txt')
    test_list = load_file_list('testlist03.txt')
    
    train_labels = extract_labels(train_list)
    test_labels = extract_labels(test_list)
    
    # Count labels
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    # Prepare data for plotting
    labels = LABELS
    train_values = [train_counts.get(label, 0) for label in labels]
    test_values = [test_counts.get(label, 0) for label in labels]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Train set distribution
    bars1 = ax1.bar(labels, train_values, color='steelblue', alpha=0.8)
    ax1.set_title('Training Set Class Distribution', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Trick Class', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Test set distribution
    bars2 = ax2.bar(labels, test_values, color='coral', alpha=0.8)
    ax2.set_title('Test Set Class Distribution', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Trick Class', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Combined comparison chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='coral', alpha=0.8)
    
    ax.set_title('Class Distribution Comparison: Train vs Test', fontsize=16, fontweight='bold')
    ax.set_xlabel('Trick Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'class_distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_class_imbalance():
    """Plot class imbalance ratios."""
    print("Generating class imbalance chart...")
    
    train_list = load_file_list('trainlist03.txt')
    train_labels = extract_labels(train_list)
    train_counts = Counter(train_labels)
    
    # Calculate imbalance ratio (relative to most common class)
    max_count = max(train_counts.values())
    labels = LABELS
    counts = [train_counts.get(label, 0) for label in labels]
    ratios = [count / max_count if count > 0 else 0 for count in counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['red' if r < 0.3 else 'orange' if r < 0.6 else 'green' for r in ratios]
    bars = ax.bar(labels, ratios, color=colors, alpha=0.7)
    
    ax.set_title('Class Imbalance Ratios (Relative to Largest Class)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Trick Class', fontsize=12)
    ax.set_ylabel('Ratio to Largest Class', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    
    # Add value labels
    for bar, ratio, count in zip(bars, ratios, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.2f}\n(n={count})',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'class_imbalance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_training_history(history_file='training_history.json'):
    """Plot training history if available."""
    history_path = os.path.join(BASE_PATH, history_file)
    
    if not os.path.exists(history_path):
        print(f"Training history file not found: {history_path}")
        print("Skipping training history charts.")
        return
    
    print("Generating training history charts...")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Plot accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Accuracy plot
    ax1.plot(epochs, history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Loss plot
    ax2.plot(epochs, history['loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax2.set_title('Model Loss Over Epochs', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Overfitting analysis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    train_acc = np.array(history['accuracy'])
    val_acc = np.array(history['val_accuracy'])
    overfitting_gap = train_acc - val_acc
    
    ax.plot(epochs, overfitting_gap, 'purple', linewidth=2, marker='o', markersize=6)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='No Overfitting')
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate Overfitting')
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Severe Overfitting')
    ax.fill_between(epochs, 0, overfitting_gap, alpha=0.3, color='purple')
    
    ax.set_title('Overfitting Analysis (Train Acc - Val Acc)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy Gap', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'overfitting_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_binary_comparison():
    """Compare 7-class vs binary classifier performance."""
    print("Generating model comparison chart...")
    
    # Sample data - you can update these with actual results
    models = ['7-Class\nClassifier', 'Binary\n(Ollie vs Kickflip)', 'Random\nGuessing']
    accuracies = [0.527, 0.536, 0.50]  # Update with actual values
    colors = ['steelblue', 'coral', 'gray']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline (50%)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.1%}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_missing_files():
    """Visualize missing .npy files."""
    print("Analyzing missing .npy files...")
    
    train_list = load_file_list('trainlist03.txt')
    test_list = load_file_list('testlist03.txt')
    all_files = train_list + test_list
    
    # Check which files are missing
    missing_by_class = {label: 0 for label in LABELS}
    existing_by_class = {label: 0 for label in LABELS}
    
    for filepath in all_files:
        label = os.path.basename(os.path.dirname(filepath))
        npy_path = os.path.join(BASE_PATH, 'Tricks', filepath.replace('.mov', '.npy'))
        
        if os.path.exists(npy_path):
            existing_by_class[label] += 1
        else:
            missing_by_class[label] += 1
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    labels = LABELS
    missing = [missing_by_class[label] for label in labels]
    existing = [existing_by_class[label] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.6
    
    bars1 = ax.bar(x, existing, width, label='Existing Files', color='green', alpha=0.7)
    bars2 = ax.bar(x, missing, width, bottom=existing, label='Missing Files', color='red', alpha=0.7)
    
    ax.set_title('Feature File Availability by Class', fontsize=16, fontweight='bold')
    ax.set_xlabel('Trick Class', fontsize=12)
    ax.set_ylabel('Number of Files', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (e, m) in enumerate(zip(existing, missing)):
        total = e + m
        if e > 0:
            ax.text(i, e/2, f'{e}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if m > 0:
            ax.text(i, e + m/2, f'{m}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(i, total + 2, f'{total}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'missing_files.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Print summary
    total_missing = sum(missing)
    total_files = sum(existing) + total_missing
    print(f"\nMissing Files Summary:")
    print(f"Total files: {total_files}")
    print(f"Existing: {sum(existing)}")
    print(f"Missing: {total_missing} ({total_missing/total_files*100:.1f}%)")


def main():
    """Generate all charts."""
    print("=" * 60)
    print("SkateboardML - Chart Generation Script")
    print("=" * 60)
    print()
    
    # Generate all charts
    plot_class_distribution()
    print()
    
    plot_class_imbalance()
    print()
    
    plot_missing_files()
    print()
    
    plot_binary_comparison()
    print()
    
    plot_training_history()
    print()
    
    print("=" * 60)
    print(f"All charts saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
