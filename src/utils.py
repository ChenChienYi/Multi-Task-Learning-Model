
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import matplotlib.pyplot as plt


def plot_training_curves(title, data):

  plt.figure(figsize=(18,7)) # Adjust figure size for a single column layout

  # Get the actual number of epochs that were run
  completed_epochs = len(data['train_losses'])

  # Plot Total Loss
  plt.subplot(1, 3, 1)
  plt.plot(range(1, completed_epochs + 1), data['train_losses'], label='Train Total Loss')
  plt.plot(range(1, completed_epochs + 1), data['val_losses'], label='Val Total Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Total Loss per Epoch')
  plt.legend()

  # Plot Attribute Loss
  plt.subplot(1, 3, 2)
  plt.plot(range(1, completed_epochs + 1), data['train_attr_losses'], label='Train Attr Loss')
  plt.plot(range(1, completed_epochs + 1), data['val_attr_losses'], label='Val Attr Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Attr Loss per Epoch')
  plt.legend()

  # Plot Landmark Loss
  plt.subplot(1, 3, 3)
  plt.plot(range(1, completed_epochs + 1), data['train_landmark_losses'], label='Train Landmark Loss')
  plt.plot(range(1, completed_epochs + 1), data['val_landmark_losses'], label='Val Landmark Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title(' Landmark Loss per Epoch')
  plt.legend()

  plt.tight_layout()
  plt.suptitle('training resulte of '+title)
  plt.show()


def evaluate_attributes(y_true_attr: np.ndarray, y_pred_attr_logits: np.ndarray, threshold: float = 0.5):
    """
    Evaluates attribute recognition performance.

    Args:
        y_true_attr (np.ndarray): Ground truth binary attribute labels.
                                  Shape: (num_samples, num_attributes).
        y_pred_attr_logits (np.ndarray): Predicted attribute logits (raw outputs from the model's head).
                                       Shape: (num_samples, num_attributes).
        threshold (float): Threshold to convert logits/probabilities to binary predictions.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    if not isinstance(y_true_attr, np.ndarray):
        y_true_attr = y_true_attr.cpu().numpy()
    if not isinstance(y_pred_attr_logits, np.ndarray):
        y_pred_attr_logits = y_pred_attr_logits.cpu().numpy()

    y_pred_attr_proba = 1 / (1 + np.exp(-y_pred_attr_logits))
    y_pred_attr_binary = (y_pred_attr_proba >= threshold).astype(int)

    num_attributes = y_true_attr.shape[1]

    per_attribute_accuracy = []
    per_attribute_f1 = []
    per_attribute_precision = []
    per_attribute_recall = []

    for i in range(num_attributes):
        true_labels = y_true_attr[:, i]
        pred_labels = y_pred_attr_binary[:, i]

        per_attribute_accuracy.append(accuracy_score(true_labels, pred_labels))
        per_attribute_f1.append(f1_score(true_labels, pred_labels, zero_division=0))
        per_attribute_precision.append(precision_score(true_labels, pred_labels, zero_division=0))
        per_attribute_recall.append(recall_score(true_labels, pred_labels, zero_division=0))

    macro_accuracy = np.mean(per_attribute_accuracy)
    macro_f1 = np.mean(per_attribute_f1)
    macro_precision = np.mean(per_attribute_precision)
    macro_recall = np.mean(per_attribute_recall)

    micro_f1 = f1_score(y_true_attr.flatten(), y_pred_attr_binary.flatten(), zero_division=0)
    micro_precision = precision_score(y_true_attr.flatten(), y_pred_attr_binary.flatten(), zero_division=0)
    micro_recall = recall_score(y_true_attr.flatten(), y_pred_attr_binary.flatten(), zero_division=0)

    metrics = {
        "macro_accuracy": macro_accuracy,
        "macro_f1_score": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f1_score": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "per_attribute_accuracy": per_attribute_accuracy,
        "per_attribute_f1_score": per_attribute_f1
    }
    return metrics

def evaluate_landmarks(y_true_landmarks: np.ndarray, y_pred_landmarks: np.ndarray):
    """
    Evaluates landmark localization performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE).

    Args:
        y_true_landmarks (np.ndarray): Ground truth landmark coordinates.
                                      Shape: (num_samples, num_landmarks * 2).
        y_pred_landmarks (np.ndarray): Predicted landmark coordinates.
                                      Shape: (num_samples, num_landmarks * 2).

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    if not isinstance(y_true_landmarks, np.ndarray):
        y_true_landmarks = y_true_landmarks.cpu().numpy()
    if not isinstance(y_pred_landmarks, np.ndarray):
        y_pred_landmarks = y_pred_landmarks.cpu().numpy()

    mae = np.mean(np.abs(y_true_landmarks - y_pred_landmarks))
    mse = np.mean((y_true_landmarks - y_pred_landmarks)**2)
    rmse = np.sqrt(mse)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }
    return metrics