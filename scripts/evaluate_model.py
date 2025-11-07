import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.paths import config, LABELS, SEQUENCE_LENGTH
import train

MODEL_PATH = str(config.BEST_MODEL)
if not os.path.exists(MODEL_PATH):
    print(f"Model {MODEL_PATH} not found. Train and create the model first (run train.py).")
    exit(1)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test data
with open(str(config.TEST_LIST)) as f:
    test_list = [row.strip() for row in list(f)]

# Create validation dataset using the same generator from train.py
valid_dataset = tf.data.Dataset.from_generator(
    train.make_generator(test_list),
    output_types=(tf.float32, tf.int32),
    output_shapes=((SEQUENCE_LENGTH, 2048), ())
)
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Collect true labels and predictions
true_labels = []
pred_labels = []

for x_batch, y_batch in valid_dataset:
    preds = model.predict(x_batch)
    pred_idx = np.argmax(preds, axis=1)
    true_labels.extend(y_batch.numpy().tolist())
    pred_labels.extend(pred_idx.tolist())

label_names = LABELS

print('Classification Report:')
print(classification_report(true_labels, pred_labels, target_names=label_names, zero_division=0))

print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))
