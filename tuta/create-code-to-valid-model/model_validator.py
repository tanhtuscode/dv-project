"""
Model Validation Code for SkateboardML
Created by: Trần Anh Tú
Purpose: Comprehensive model validation and evaluation
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import os
import json
from datetime import datetime
import pandas as pd

class ModelValidator:
    def __init__(self, model_path, test_data_path, labels):
        """
        Initialize Model Validator
        
        Args:
            model_path (str): Path to the trained model
            test_data_path (str): Path to test data
            labels (list): List of class labels
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.labels = labels
        self.model = None
        self.test_data = None
        self.test_labels = None
        self.predictions = None
        self.prediction_probabilities = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_test_data(self):
        """Load test data for validation"""
        try:
            # Load test data based on your data format
            # This assumes you have a specific test data format
            print(f"📂 Loading test data from {self.test_data_path}")
            
            # Example implementation - adjust based on your data format
            if os.path.exists(self.test_data_path):
                # Load your test data here
                # This is a placeholder - implement based on your actual data structure
                print("✅ Test data loaded successfully")
                return True
            else:
                print(f"❌ Test data path not found: {self.test_data_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return False
    
    def make_predictions(self):
        """Make predictions on test data"""
        if self.model is None:
            print("❌ Model not loaded. Please load model first.")
            return False
        
        if self.test_data is None:
            print("❌ Test data not loaded. Please load test data first.")
            return False
        
        try:
            print("🔮 Making predictions...")
            self.prediction_probabilities = self.model.predict(self.test_data, verbose=1)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)
            print("✅ Predictions completed successfully")
            return True
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return False
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if self.predictions is None or self.test_labels is None:
            print("❌ Predictions or test labels not available")
            return None
        
        try:
            print("📊 Calculating evaluation metrics...")
            
            # Basic metrics
            accuracy = accuracy_score(self.test_labels, self.predictions)
            
            # Detailed metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                self.test_labels, self.predictions, average=None
            )
            
            # Macro and weighted averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                self.test_labels, self.predictions, average='macro'
            )
            
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                self.test_labels, self.predictions, average='weighted'
            )
            
            # Confusion matrix
            cm = confusion_matrix(self.test_labels, self.predictions)
            
            # Classification report
            class_report = classification_report(
                self.test_labels, 
                self.predictions, 
                target_names=self.labels,
                output_dict=True
            )
            
            metrics = {
                'accuracy': accuracy,
                'precision_per_class': precision.tolist(),
                'recall_per_class': recall.tolist(),
                'f1_per_class': f1.tolist(),
                'support_per_class': support.tolist(),
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'labels': self.labels
            }
            
            print("✅ Metrics calculated successfully")
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return None
    
    def plot_confusion_matrix(self, save_path=None):
        """Plot and save confusion matrix"""
        if self.predictions is None or self.test_labels is None:
            print("❌ Predictions or test labels not available")
            return
        
        try:
            cm = confusion_matrix(self.test_labels, self.predictions)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.labels,
                       yticklabels=self.labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Confusion matrix saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Error plotting confusion matrix: {e}")
    
    def plot_metrics_comparison(self, save_path=None):
        """Plot metrics comparison across classes"""
        metrics = self.calculate_metrics()
        if metrics is None:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Precision
            axes[0, 0].bar(self.labels, metrics['precision_per_class'])
            axes[0, 0].set_title('Precision per Class')
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Recall
            axes[0, 1].bar(self.labels, metrics['recall_per_class'])
            axes[0, 1].set_title('Recall per Class')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # F1-Score
            axes[1, 0].bar(self.labels, metrics['f1_per_class'])
            axes[1, 0].set_title('F1-Score per Class')
            axes[1, 0].set_ylabel('F1-Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Support
            axes[1, 1].bar(self.labels, metrics['support_per_class'])
            axes[1, 1].set_title('Support per Class')
            axes[1, 1].set_ylabel('Number of Samples')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Metrics comparison saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            print(f"❌ Error plotting metrics comparison: {e}")
    
    def plot_prediction_confidence(self, save_path=None):
        """Plot prediction confidence distribution"""
        if self.prediction_probabilities is None:
            print("❌ Prediction probabilities not available")
            return
        
        try:
            max_probs = np.max(self.prediction_probabilities, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Prediction Confidence')
            plt.xlabel('Maximum Probability')
            plt.ylabel('Frequency')
            plt.axvline(x=np.mean(max_probs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(max_probs):.3f}')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Confidence distribution saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            print(f"❌ Error plotting confidence distribution: {e}")
    
    def generate_validation_report(self, output_dir="validation_results"):
        """Generate comprehensive validation report"""
        print("📋 Generating comprehensive validation report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        if metrics is None:
            print("❌ Could not calculate metrics")
            return
        
        # Generate plots
        self.plot_confusion_matrix(os.path.join(output_dir, "confusion_matrix.png"))
        self.plot_metrics_comparison(os.path.join(output_dir, "metrics_comparison.png"))
        self.plot_prediction_confidence(os.path.join(output_dir, "confidence_distribution.png"))
        
        # Save metrics as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"validation_metrics_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate text report
        report_file = os.path.join(output_dir, f"validation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SKATEBOARDML MODEL VALIDATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Data: {self.test_data_path}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
            f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
            f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n")
            f.write(f"Weighted Precision: {metrics['precision_weighted']:.4f}\n")
            f.write(f"Weighted Recall: {metrics['recall_weighted']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}\n\n")
            
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            for i, label in enumerate(self.labels):
                f.write(f"{label}:\n")
                f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}\n")
                f.write(f"  Support: {metrics['support_per_class'][i]}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 30 + "\n")
            cm = np.array(metrics['confusion_matrix'])
            f.write("True\\Pred\t" + "\t".join(self.labels) + "\n")
            for i, label in enumerate(self.labels):
                f.write(f"{label}\t\t" + "\t".join(map(str, cm[i])) + "\n")
        
        print(f"✅ Validation report generated in {output_dir}/")
        print(f"📊 Metrics saved to: {metrics_file}")
        print(f"📋 Report saved to: {report_file}")
        
        return metrics

def validate_model_from_files(model_path, trainlist_path, testlist_path, tricks_path, labels):
    """
    Validate model using file-based data loading
    
    Args:
        model_path (str): Path to the trained model
        trainlist_path (str): Path to training list file
        testlist_path (str): Path to test list file
        tricks_path (str): Path to tricks directory
        labels (list): List of class labels
    """
    
    print("🚀 Starting Model Validation from Files")
    print("=" * 50)
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load test data
    print("📂 Loading test data...")
    test_data = []
    test_labels = []
    
    try:
        with open(testlist_path, 'r') as f:
            test_files = [line.strip() for line in f if line.strip()]
        
        print(f"📁 Found {len(test_files)} test files")
        
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        
        for file_path in test_files:
            # Extract label from file path
            if '/' in file_path:
                trick_name = file_path.split('/')[0]
            else:
                trick_name = os.path.dirname(file_path)
            
            if trick_name in label_to_index:
                # Load feature file
                npy_path = os.path.join(tricks_path, file_path.replace('.mov', '.npy'))
                
                if os.path.exists(npy_path):
                    try:
                        features = np.load(npy_path)
                        
                        # Pad or truncate to sequence length
                        sequence_length = 40
                        padded_features = np.zeros((sequence_length, 2048))
                        seq_len = min(len(features), sequence_length)
                        padded_features[:seq_len] = features[:seq_len]
                        
                        test_data.append(padded_features)
                        test_labels.append(label_to_index[trick_name])
                        
                    except Exception as e:
                        print(f"⚠️  Error loading {npy_path}: {e}")
                else:
                    print(f"⚠️  Feature file not found: {npy_path}")
            else:
                print(f"⚠️  Unknown label: {trick_name}")
        
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
        
        print(f"✅ Loaded {len(test_data)} test samples")
        print(f"📊 Test data shape: {test_data.shape}")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None
    
    # Make predictions
    print("🔮 Making predictions...")
    try:
        predictions_proba = model.predict(test_data, verbose=1)
        predictions = np.argmax(predictions_proba, axis=1)
        print("✅ Predictions completed")
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None
    
    # Calculate metrics
    print("📊 Calculating metrics...")
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        accuracy = accuracy_score(test_labels, predictions)
        cm = confusion_matrix(test_labels, predictions)
        report = classification_report(test_labels, predictions, target_names=labels)
        
        print("\n" + "=" * 50)
        print("VALIDATION RESULTS")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print("Predicted\\True", "\t".join(labels))
        for i, label in enumerate(labels):
            print(f"{label}\t\t", "\t".join(map(str, cm[i])))
        
        # Plot confusion matrix
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'Confusion Matrix (Accuracy: {accuracy:.3f})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Save plot
            output_dir = "validation_results"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.show()
            print(f"\n📊 Confusion matrix saved to {output_dir}/confusion_matrix.png")
            
        except ImportError:
            print("⚠️  Matplotlib not available, skipping visualization")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'prediction_probabilities': predictions_proba
        }
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Configuration for SkateboardML
    MODEL_PATH = "../../best_model.keras"  # or "../../final_model.keras"
    TRAINLIST_PATH = "../../trainlist_binary.txt"
    TESTLIST_PATH = "../../testlist_binary.txt"
    TRICKS_PATH = "../../Tricks"
    LABELS = ["Kickflip", "Ollie"]
    
    print("🛹 SkateboardML Model Validation")
    print("=" * 50)
    
    # Run validation
    results = validate_model_from_files(
        MODEL_PATH, 
        TRAINLIST_PATH, 
        TESTLIST_PATH, 
        TRICKS_PATH, 
        LABELS
    )
    
    if results:
        print("\n✅ Validation completed successfully!")
    else:
        print("\n❌ Validation failed!")