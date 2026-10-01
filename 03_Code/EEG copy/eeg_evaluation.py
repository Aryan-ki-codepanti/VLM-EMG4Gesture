
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def collect_predictions_and_labels(model, dataloader, device):
    """
    Collects true labels and model predictions from a DataLoader.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the evaluation set.
        device (torch.device): The device to run the model on.

    Returns:
        tuple: (all_true_labels, all_pred_labels) as numpy arrays.
    """
    model.eval() # Set model to evaluation mode
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad(): # Disable gradient calculation for inference
        for eeg_data, labels, _ in dataloader:
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)

            outputs = model(eeg_data)
            _, predicted = torch.max(outputs.data, 1) # Get the class with the highest score

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

    return np.array(all_true_labels), np.array(all_pred_labels)

def calculate_metrics(true_labels, pred_labels, num_classes):
    """
    Calculates classification metrics: accuracy, macro F1, per-class P/R/F1, and confusion matrix.

    Args:
        true_labels (np.ndarray): Array of true class labels.
        pred_labels (np.ndarray): Array of predicted class labels.
        num_classes (int): Total number of classes.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    report = {}

    # Overall Accuracy
    report['overall_accuracy'] = accuracy_score(true_labels, pred_labels)

    # Macro F1-score
    report['macro_f1'] = f1_score(true_labels, pred_labels, average='macro')

    # Per-class Precision, Recall, F1-score
    # labels argument ensures all classes are covered even if not present in pred/true
    target_names = [f'Gesture {i+1}' for i in range(num_classes)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=np.arange(num_classes), average=None)

    report['per_class_precision'] = precision.tolist()
    report['per_class_recall'] = recall.tolist()
    report['per_class_f1'] = f1.tolist()
    report['per_class_names'] = target_names

    # Confusion Matrix
    report['confusion_matrix'] = confusion_matrix(true_labels, pred_labels, labels=np.arange(num_classes)).tolist()

    return report

def evaluate_model(model, dataloader, num_classes, device):
    """
    Main function to evaluate the model and generate a comprehensive report.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        num_classes (int): Total number of classes.
        device (torch.device): The device to run the model on.

    Returns:
        dict: A dictionary containing the full evaluation report.
    """
    print(f"\nEvaluating model on {len(dataloader.dataset)} samples...")
    true_labels, pred_labels = collect_predictions_and_labels(model, dataloader, device)
    metrics = calculate_metrics(true_labels, pred_labels, num_classes)
    print("Evaluation complete.")
    return metrics
