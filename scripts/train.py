import tensorflow as tf
import os
import numpy as np
import sys
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelBinarizer

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import config, SEQUENCE_LENGTH, LABELS

# Dynamic paths that work on any PC
BASE_PATH = str(config.TRICKS_DIR)
TRAIN_LIST_PATH = str(config.TRAIN_LIST)
TEST_LIST_PATH = str(config.TEST_LIST)
MODELS_DIR = str(config.MODELS_DIR)

# Map labels to integer indices for sparse labels and class weights
label_to_index = {label: idx for idx, label in enumerate(LABELS)}

def build_model(num_classes):
    """Build LSTM model for skateboard trick classification.
    
    Improved architecture to reduce overfitting:
    - Smaller LSTM (128 units instead of 512)
    - Less aggressive dropout (0.3 instead of 0.5)
    - L2 regularization to prevent overfitting
    - Simpler architecture for small datasets
    """
    from tensorflow.keras import regularizers
    
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0.),
        
        # Smaller LSTM with regularization
        tf.keras.layers.LSTM(
            128,  # Reduced from 512
            dropout=0.3,  # Reduced from 0.5
            recurrent_dropout=0.3,  # Reduced from 0.5
            kernel_regularizer=regularizers.l2(0.01),
            recurrent_regularizer=regularizers.l2(0.01)
        ),
        
        # Smaller dense layer
        tf.keras.layers.Dense(
            64,  # Reduced from 256
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        ),
        
        # Less dropout
        tf.keras.layers.Dropout(0.3),  # Reduced from 0.5
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use a lower learning rate for better convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

def make_generator(file_list):
    """Generator that yields (sequence, label_index) where label_index is an int.
    Using integer labels allows us to pass `class_weight` to model.fit.
    """
    def generator():
        file_list_copy = file_list.copy()
        np.random.shuffle(file_list_copy)
        for path in file_list_copy:
            # Use cross-platform path handling
            full_path = os.path.join(BASE_PATH, path).replace('.mov', '.npy')

            # Check if file exists
            if not os.path.exists(full_path):
                print(f"Warning: File not found: {full_path}")
                continue

            label = os.path.basename(os.path.dirname(path))
            try:
                features = np.load(full_path)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue

            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048), dtype=np.float32)
            padded_sequence[0:len(features)] = np.array(features)

            # Get integer label
            label_idx = label_to_index.get(label, None)
            if label_idx is None:
                print(f"Unknown label '{label}' for path {path}")
                continue

            yield padded_sequence, np.int32(label_idx)
    return generator

def compute_class_weights(train_list, label_to_index):
    """Compute class weights to handle imbalanced datasets."""
    train_counts = Counter([p.split('/')[0] for p in train_list])
    total = sum(train_counts.values())
    class_weight = {}
    
    for label, idx in label_to_index.items():
        count = train_counts.get(label, 0)
        # Inverse frequency weighting
        class_weight[idx] = (total / (len(label_to_index) * count)) if count > 0 else 1.0
    
    return class_weight

def run_training(epochs=30, use_class_weights=True):
    """
    Train the skateboard trick classifier.
    
    Args:
        epochs: Number of training epochs (increased to 30 for better learning)
        use_class_weights: Whether to use class weights to handle imbalance
    """
    print("\n" + "="*60)
    print("SkateboardML Training Script")
    print("="*60)
    print(f"Labels: {LABELS}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"Training file: {TRAIN_LIST_PATH}")
    print(f"Test file: {TEST_LIST_PATH}")
    print("="*60 + "\n")
    
    # Load train and test lists
    with open(TEST_LIST_PATH) as f:
        test_list = [row.strip() for row in list(f)]

    with open(TRAIN_LIST_PATH) as f:
        train_list = [row.strip() for row in list(f)]
        train_list = [row.split(' ')[0] for row in train_list]

    print(f"Training samples: {len(train_list)}")
    print(f"Test samples: {len(test_list)}")
    print()
    
    # Compute class weights
    class_weight = None
    if use_class_weights:
        class_weight = compute_class_weights(train_list, label_to_index)
        print("Class weights (to handle imbalance):")
        for label, idx in label_to_index.items():
            print(f"  {label}: {class_weight[idx]:.3f}")
        print()
    
    # Build model
    model = build_model(num_classes=len(LABELS))
    print("Model architecture:")
    model.summary()
    print()
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_generator(
        make_generator(train_list),
        output_types=(tf.float32, tf.int32),
        output_shapes=((SEQUENCE_LENGTH, 2048), ())
    )
    train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_generator(
        make_generator(test_list),
        output_types=(tf.float32, tf.int32),
        output_shapes=((SEQUENCE_LENGTH, 2048), ())
    )
    valid_dataset = valid_dataset.batch(16).prefetch(tf.data.AUTOTUNE)
    
    # Create output directories
    log_dir = config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup callbacks with improved early stopping
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir), 
        update_freq=1000
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        str(config.BEST_MODEL), 
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Improved early stopping - more patience, monitor val_loss
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased from 5 to 10
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001  # Only stop if improvement is less than 0.001
    )
    
    # Reduce learning rate when stuck
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=5,  # After 5 epochs without improvement
        min_lr=0.00001,
        verbose=1
    )
    
    # Train the model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = model.fit(
        train_dataset, 
        epochs=epochs, 
        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping, reduce_lr], 
        validation_data=valid_dataset,
        class_weight=class_weight
    )
    
    # Save the final model
    final_model_path = str(config.FINAL_MODEL)
    model.save(final_model_path)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Final model saved: {final_model_path}")
    print(f"Best model saved: {config.BEST_MODEL}")
    print(f"Logs saved to: {log_dir}")
    print("="*60)
    
    # Save training history for visualization
    history_path = config.PROJECT_ROOT / 'training_history.json'
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    }
    
    with open(history_path, 'w') as f:
        import json
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {history_path}")
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"  Final training accuracy:   {history.history['accuracy'][-1]:.4f}")
    print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Final training loss:       {history.history['loss'][-1]:.4f}")
    print(f"  Final validation loss:     {history.history['val_loss'][-1]:.4f}")
    print(f"  Best validation accuracy:  {max(history.history['val_accuracy']):.4f}")
    print()
    
    return history

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SkateboardML classifier')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--no-class-weights', action='store_true', 
                       help='Disable class weights')
    
    args = parser.parse_args()
    
    run_training(
        epochs=args.epochs,
        use_class_weights=not args.no_class_weights
    )
