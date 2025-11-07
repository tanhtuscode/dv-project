# -*- coding: utf-8 -*-
"""
Windows-compatible training script for SkateboardML
"""

import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# Update BASE_PATH for Windows
BASE_PATH = 'd:/DV/SkateboardML/Tricks'
SEQUENCE_LENGTH = 40

# Labels for all tricks
LABELS = ["Back180", "Front180", "Frontshuvit", "Kickflip", "Ollie", "Shuvit", "Varial"]
# map labels to integer indices so we can use sparse labels and class weights
label_to_index = {label: idx for idx, label in enumerate(LABELS)}

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Load train and test lists
with open('testlist03.txt') as f:
    test_list = [row.strip() for row in list(f)]

with open('trainlist03.txt') as f:
    train_list = [row.strip() for row in list(f)]
    train_list = [row.split(' ')[0] for row in train_list]

print(f"Training samples: {len(train_list)}")
print(f"Test samples: {len(test_list)}")

def make_generator(file_list):
    """Generator that yields (sequence, label_index) where label_index is an int.
    Using integer labels allows us to pass `class_weight` to model.fit.
    """
    def generator():
        np.random.shuffle(file_list)
        for path in file_list:
            # Use Windows-compatible path separator
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

            # integer label
            label_idx = label_to_index.get(label, None)
            if label_idx is None:
                print(f"Unknown label '{label}' for path {path}")
                continue

            yield padded_sequence, np.int32(label_idx)
    return generator

# Create datasets
train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                 output_types=(tf.float32, tf.int32),
                 output_shapes=((SEQUENCE_LENGTH, 2048), ()))
train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int32),
                 output_shapes=((SEQUENCE_LENGTH, 2048), ()))
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Create log directory if it doesn't exist
if not os.path.exists('log'):
    os.makedirs('log')

# Setup callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras', 
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

def run_training(epochs=17):
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    # compute class weights from train_list counts
    from collections import Counter
    train_counts = Counter([p.split('/')[0] for p in train_list])
    total = sum(train_counts.values())
    class_weight = {}
    for label, idx in label_to_index.items():
        # inverse frequency
        count = train_counts.get(label, 0)
        # avoid division by zero
        class_weight[idx] = (total / (len(LABELS) * count)) if count>0 else 1.0

    print('Class weights:', class_weight)

    # Train the model (pass class_weight for sparse labels)
    history = model.fit(
        train_dataset, 
        epochs=epochs, 
        callbacks=[tensorboard_callback, checkpoint_callback], 
        validation_data=valid_dataset,
        class_weight=class_weight
    )

    # Save the final model
    model.save('final_model.keras')
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Final model saved as: final_model.keras")
    print(f"Best model saved as: best_model.keras")
    print("="*50)

    # Print summary of training
    print("\nTraining Summary:")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")


if __name__ == '__main__':
    run_training()
