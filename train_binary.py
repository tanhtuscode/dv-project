# -*- coding: utf-8 -*-
"""
Binary classifier for Ollie vs Kickflip
"""

import tensorflow as tf
import os
import numpy as np
from collections import Counter

# Update BASE_PATH for Windows
BASE_PATH = 'd:/DV/SkateboardML/Tricks'
SEQUENCE_LENGTH = 40

# Binary classification: Ollie vs Kickflip only
LABELS = ["Kickflip", "Ollie"]
label_to_index = {label: idx for idx, label in enumerate(LABELS)}

# Build the LSTM model (simpler since it's binary)
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.LSTM(256, dropout=0.4, recurrent_dropout=0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Load train and test lists, filter for Ollie and Kickflip only
with open('testlist03.txt') as f:
    test_list = [row.strip() for row in list(f) if row.strip()]
    test_list = [p for p in test_list if any(label in p for label in LABELS)]

with open('trainlist03.txt') as f:
    train_list = [row.strip() for row in list(f) if row.strip()]
    train_list = [row.split(' ')[0] for row in train_list]
    train_list = [p for p in train_list if any(label in p for label in LABELS)]

print(f"Training samples: {len(train_list)}")
print(f"Test samples: {len(test_list)}")

# Count per class
train_counts = Counter([p.split('/')[0] for p in train_list])
test_counts = Counter([p.split('/')[0] for p in test_list])
print(f"\nTrain distribution: {dict(train_counts)}")
print(f"Test distribution: {dict(test_counts)}")

def make_generator(file_list):
    """Generator that yields (sequence, label_index) where label_index is an int."""
    def generator():
        np.random.shuffle(file_list)
        for path in file_list:
            # Use Windows-compatible path separator
            full_path = os.path.join(BASE_PATH, path).replace('.mov', '.npy')

            # Check if file exists
            if not os.path.exists(full_path):
                continue

            label = os.path.basename(os.path.dirname(path))
            
            # Skip if not in our binary labels
            if label not in LABELS:
                continue
                
            try:
                features = np.load(full_path)
            except Exception as e:
                continue

            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048), dtype=np.float32)
            padded_sequence[0:len(features)] = np.array(features)

            # integer label
            label_idx = label_to_index[label]
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
if not os.path.exists('log_binary'):
    os.makedirs('log_binary')

# Setup callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log_binary', update_freq=1000)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    'best_binary_model.keras', 
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n" + "="*50)
print("Starting binary training (Ollie vs Kickflip)...")
print("="*50 + "\n")

# Train the model
history = model.fit(
    train_dataset, 
    epochs=30, 
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping], 
    validation_data=valid_dataset
)

# Save the final model
model.save('final_binary_model.keras')
print("\n" + "="*50)
print("Training complete!")
print(f"Final model saved as: final_binary_model.keras")
print(f"Best model saved as: best_binary_model.keras")
print("="*50)

# Print summary of training
print("\nTraining Summary:")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Evaluate on test set
print("\n" + "="*50)
print("Evaluating on test set...")
print("="*50)
test_loss, test_acc = model.evaluate(valid_dataset)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
