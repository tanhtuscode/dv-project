import numpy as np
import os
import train_windows
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = 'best_model.keras'
if not os.path.exists(MODEL_PATH):
    print(f"Model {MODEL_PATH} not found. Train and create the model first (run train_windows.py).")
    exit(1)

model = train_windows.tf.keras.models.load_model(MODEL_PATH)

# collect true labels and predictions
true_labels = []
pred_labels = []

for x_batch, y_batch in train_windows.valid_dataset:
    preds = model.predict(x_batch)
    pred_idx = np.argmax(preds, axis=1)
    true_labels.extend(y_batch.numpy().tolist())
    pred_labels.extend(pred_idx.tolist())

label_names = train_windows.LABELS

print('Classification Report:')
print(classification_report(true_labels, pred_labels, target_names=label_names, zero_division=0))

print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))
