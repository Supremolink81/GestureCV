import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def print_evaluation_metrics(ground_truth_array: np.array, model_prediction_array: np.array, labels: list[str]) -> None:

    """
    Prints various evaluation metrics to the console, including:

    - confusion matrix
    - accuracy
    - precision
    - recall
    - f1 score

    And does so according to given class labels.

    Args:

        `np.array ground_truth_array`: the array to use for ground truth labels.

        `np.array model_prediction_array`: the array to use for model predictions.

        `list[str] labels`: the labels to use for printing.

    Returns:

        None
    """

    overall_accuracy: float = np.sum(ground_truth_array == model_prediction_array)[0] / ground_truth_array.shape[0]

    confusion_matrix_of_results: np.array = confusion_matrix(ground_truth_array, model_prediction_array)

    precision_array, recall_array, f1_score_array, _ = precision_recall_fscore_support(ground_truth_array, model_prediction_array)

    print(f"Overall accuracy: {overall_accuracy}")

    for label_index, label in enumerate(labels):

        print(f"Statistics for label {label}:\n")

        print(f"Precision: {recall_array[label_index]}")

        print(f"Recall: {recall_array[label_index]}")

        print(f"F1 score: {f1_score_array[label_index]}\n")