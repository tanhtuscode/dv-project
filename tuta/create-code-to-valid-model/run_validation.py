"""
Simple Model Validation Runner
Created by: Trần Anh Tú
Purpose: Easy-to-use model validation for SkateboardML
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model_validator import validate_model_from_files
import argparse

def main():
    """Main validation function"""
    
    # Default paths (adjust as needed)
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    DEFAULT_MODEL = os.path.join(BASE_PATH, "best_model.keras")
    DEFAULT_TRAINLIST = os.path.join(BASE_PATH, "trainlist_binary.txt")
    DEFAULT_TESTLIST = os.path.join(BASE_PATH, "testlist_binary.txt")
    DEFAULT_TRICKS = os.path.join(BASE_PATH, "Tricks")
    DEFAULT_LABELS = ["Kickflip", "Ollie"]
    
    parser = argparse.ArgumentParser(description='Validate SkateboardML Model')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Path to model file')
    parser.add_argument('--trainlist', default=DEFAULT_TRAINLIST, help='Path to training list')
    parser.add_argument('--testlist', default=DEFAULT_TESTLIST, help='Path to test list')
    parser.add_argument('--tricks', default=DEFAULT_TRICKS, help='Path to tricks directory')
    parser.add_argument('--labels', nargs='+', default=DEFAULT_LABELS, help='List of labels')
    
    args = parser.parse_args()
    
    print("🛹 SkateboardML Model Validation")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Test List: {args.testlist}")
    print(f"Tricks Directory: {args.tricks}")
    print(f"Labels: {args.labels}")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.testlist):
        print(f"❌ Test list not found: {args.testlist}")
        return
    
    if not os.path.exists(args.tricks):
        print(f"❌ Tricks directory not found: {args.tricks}")
        return
    
    # Run validation
    results = validate_model_from_files(
        args.model,
        args.trainlist,
        args.testlist,
        args.tricks,
        args.labels
    )
    
    if results:
        print("\n🎉 Validation completed successfully!")
        print(f"📊 Final Accuracy: {results['accuracy']:.4f}")
    else:
        print("\n❌ Validation failed!")

if __name__ == "__main__":
    main()