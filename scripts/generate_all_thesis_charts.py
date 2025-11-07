"""
Master Chart Generator for SkateboardML Thesis
Generates all visualization charts required for academic thesis
Organized by priority: OPTIMAL, IMPORTANT, USEFUL, NICE-TO-HAVE
"""

import os
import sys
from pathlib import Path
import subprocess
import time

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import config

def generate_all_thesis_charts():
    """Generate all charts for the thesis"""
    
    print("\n" + "="*80)
    print("SKATEBOARDML THESIS - COMPREHENSIVE CHART GENERATION")
    print("="*80)
    print("Generating all visualization charts for academic thesis")
    print("Categories: OPTIMAL (required), IMPORTANT (recommended), USEFUL (nice-to-have)")
    print("="*80 + "\n")
    
    scripts_dir = Path(__file__).parent
    
    # Track timing
    start_time = time.time()
    
    try:
        # 1. Generate core thesis charts (OPTIMAL + IMPORTANT)
        print("🎯 STEP 1: Core Thesis Charts (OPTIMAL + IMPORTANT)")
        print("-" * 50)
        result = subprocess.run([
            sys.executable, 
            str(scripts_dir / "thesis_charts.py")
        ], capture_output=True, text=True, cwd=config.PROJECT_ROOT)
        
        if result.returncode == 0:
            print("✅ Core thesis charts generated successfully!")
        else:
            print("❌ Error generating core charts:")
            print(result.stderr)
        
        print()
        
        # 2. Generate advanced charts (USEFUL + NICE-TO-HAVE)
        print("📊 STEP 2: Advanced Analysis Charts (USEFUL)")
        print("-" * 50)
        result = subprocess.run([
            sys.executable, 
            str(scripts_dir / "simple_advanced_charts.py")
        ], capture_output=True, text=True, cwd=config.PROJECT_ROOT)
        
        if result.returncode == 0:
            print("✅ Advanced analysis charts generated successfully!")
        else:
            print("❌ Error generating advanced charts:")
            print(result.stderr)
        
        print()
        
    except Exception as e:
        print(f"❌ Error during chart generation: {e}")
    
    # Generate summary report
    end_time = time.time()
    total_time = end_time - start_time
    
    print("📋 GENERATION SUMMARY")
    print("-" * 50)
    
    # Count generated files
    thesis_charts_dir = config.OUTPUTS_DIR / "thesis_charts"
    advanced_charts_dir = config.OUTPUTS_DIR / "advanced_charts"
    
    thesis_files = list(thesis_charts_dir.glob("*.png")) if thesis_charts_dir.exists() else []
    advanced_files = list(advanced_charts_dir.glob("*.png")) if advanced_charts_dir.exists() else []
    
    print(f"📁 Core Charts Directory: {thesis_charts_dir}")
    print(f"   Generated: {len(thesis_files)} PNG files")
    for file in thesis_files:
        print(f"   ✓ {file.name}")
    
    print(f"\n📁 Advanced Charts Directory: {advanced_charts_dir}")
    print(f"   Generated: {len(advanced_files)} PNG files")
    for file in advanced_files:
        print(f"   ✓ {file.name}")
    
    print(f"\n⏱️  Total Generation Time: {total_time:.1f} seconds")
    print(f"📊 Total Charts Generated: {len(thesis_files) + len(advanced_files)}")
    
    # Create comprehensive chart index
    create_chart_index()
    
    print("\n" + "="*80)
    print("✅ ALL THESIS CHARTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("Charts are organized and ready for thesis inclusion.")
    print("See chart_index.md for detailed descriptions and usage.")
    print("="*80 + "\n")

def create_chart_index():
    """Create a comprehensive index of all generated charts"""
    
    index_content = """# SkateboardML Thesis Charts Index

## Overview
This document provides a comprehensive index of all visualization charts generated for the SkateboardML thesis.

## Chart Categories

### OPTIMAL (Required) Charts
These charts are essential and must be included in the thesis.

#### 1. Training Curves (training_curves.png)
- **Location**: Ch.3 (Training) — Figure 3.1
- **Purpose**: Monitor convergence, detect overfitting/underfitting
- **Content**: Loss and Accuracy curves for training vs validation
- **Caption**: "Loss và Accuracy (training & validation) theo epoch"
- **Technical**: Shows best validation points, 300 DPI PNG

#### 2. Confusion Matrix (confusion_matrix.png)
- **Location**: Ch.3 (Results) — Figure 3.2
- **Purpose**: Analyze classification errors between classes
- **Content**: Both absolute numbers and normalized percentages
- **Caption**: "Ma trận nhầm lẫn trên tập kiểm tra (số lượng và tỷ lệ chuẩn hoá)"
- **Technical**: Heatmap with clear row=actual, column=predicted labels

#### 3. Classification Metrics (classification_metrics.png)
- **Location**: Ch.3 (Results) — near confusion matrix
- **Purpose**: Compare precision, recall, F1-score per class
- **Content**: Bar chart with values displayed on bars
- **Caption**: "Precision, Recall, F1-score cho từng lớp"
- **Technical**: Clear value labels, grid for readability

### IMPORTANT (Recommended) Charts
These charts provide valuable insights and are recommended for inclusion.

#### 4. ROC Curve (roc_curve.png)
- **Location**: Ch.3 (Results)
- **Purpose**: Evaluate binary classification performance
- **Content**: ROC curve with AUC score
- **Caption**: "ROC curve và AUC trên tập kiểm tra"
- **Technical**: Diagonal reference line, AUC annotation

#### 5. Precision-Recall Curve (precision_recall_curve.png)
- **Location**: Ch.3 (Results)
- **Purpose**: Alternative performance metric for imbalanced data
- **Content**: PR curve with average precision score
- **Caption**: "Precision-Recall curve (test set)"
- **Technical**: AP score annotation

#### 6. Confidence Distribution (confidence_distribution.png)
- **Location**: Ch.3 (Analysis)
- **Purpose**: Analyze model confidence and calibration
- **Content**: Histogram and box plots of prediction confidence
- **Caption**: "Phân phối xác suất dự đoán cho lớp đúng và lớp sai"
- **Technical**: Separate distributions for correct/incorrect predictions

### USEFUL (Nice-to-Have) Charts
These charts add depth to the research and improve thesis quality.

#### 7. Dataset Statistics (dataset_statistics.png)
- **Location**: Ch.2 (Data Collection)
- **Purpose**: Show data distribution and balance
- **Content**: Pie charts and bar charts of class distribution
- **Caption**: "Phân bố số lượng video theo lớp và tổng quan dataset"
- **Technical**: Multiple subplots showing train/test splits

#### 8. Data Augmentation Examples (augmentation_examples.png)
- **Location**: Ch.2 (Preprocessing)
- **Purpose**: Demonstrate data augmentation techniques
- **Content**: Before/after examples of augmentation
- **Caption**: "Ví dụ các phép Data Augmentation"
- **Technical**: Grid layout showing original and augmented versions

#### 9. Training Time Analysis (training_time_analysis.png)
- **Location**: Ch.3 (Environment/Experiments)
- **Purpose**: Resource usage and scalability analysis
- **Content**: Time breakdown, memory usage, efficiency metrics
- **Caption**: "Phân tích thời gian huấn luyện và sử dụng tài nguyên"
- **Technical**: Multiple subplots with detailed metrics

#### 10. Architecture Comparison (architecture_comparison.png)
- **Location**: Ch.3 (Experiments)
- **Purpose**: Ablation study and model justification
- **Content**: Performance comparison across architectures
- **Caption**: "So sánh hiệu năng giữa các kiến trúc mô hình"
- **Technical**: Bar charts with highlighted best model

#### 11. Data Quality Analysis (data_quality_analysis.png)
- **Location**: Ch.2 (Data Collection)
- **Purpose**: Demonstrate data quality and preprocessing
- **Content**: Video length, resolution, frame rate distributions
- **Caption**: "Phân tích chất lượng và đặc tính dữ liệu"
- **Technical**: Multiple distribution plots

#### 12. Performance Over Time (performance_over_time.png)
- **Location**: Ch.3 (Training Analysis)
- **Purpose**: Detailed training progress analysis
- **Content**: Moving averages, improvement rates, overfitting analysis
- **Caption**: "Phân tích chi tiết quá trình huấn luyện"
- **Technical**: Multiple subplots with trend analysis

## File Locations
```
outputs/
├── thesis_charts/           # Core charts (OPTIMAL + IMPORTANT)
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── classification_metrics.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── confidence_distribution.png
│   └── dataset_statistics.png
│
└── advanced_charts/         # Advanced charts (USEFUL)
    ├── augmentation_examples.png
    ├── training_time_analysis.png
    ├── architecture_comparison.png
    ├── data_quality_analysis.png
    ├── performance_over_time.png
    └── hardware_specs.txt
```

## Usage Guidelines

### For Academic Thesis
1. **Include ALL OPTIMAL charts** - these are essential for academic rigor
2. **Include most IMPORTANT charts** - these demonstrate thorough analysis
3. **Select relevant USEFUL charts** - based on thesis focus and space constraints

### Figure Quality
- All charts are generated at 300 DPI for publication quality
- PNG format for compatibility with most document systems
- Consistent styling and fonts for professional appearance

### Captions and References
- Use provided Vietnamese captions or translate as needed
- Reference charts by their suggested locations (Ch.2, Ch.3, etc.)
- Include detailed discussions of what each chart reveals

### Customization
- Charts can be regenerated with different parameters
- Modify scripts in `scripts/` directory for customization
- All charts use dynamic data loading where possible

## Hardware Requirements
See `hardware_specs.txt` for detailed system requirements and performance metrics used during chart generation.

## Last Updated
Generated automatically by master chart generation script.
"""
    
    index_path = config.OUTPUTS_DIR / "chart_index.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"📋 Created comprehensive chart index: {index_path}")

if __name__ == "__main__":
    generate_all_thesis_charts()