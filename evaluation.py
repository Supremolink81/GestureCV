import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

EMPTY_MATRIX: np.array = np.array(
    5 * [5 * [0]], dtype=np.int64,
)

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

    global EMPTY_MATRIX

    overall_accuracy: float = np.sum(ground_truth_array == model_prediction_array) / ground_truth_array.shape[0]

    confusion_matrix_of_results: np.array = confusion_matrix(ground_truth_array, model_prediction_array)

    EMPTY_MATRIX += confusion_matrix_of_results

    precision_array, recall_array, f1_score_array, _ = precision_recall_fscore_support(ground_truth_array, model_prediction_array)

    print(f"Overall accuracy: {overall_accuracy}")

    print(f"Confusion Matrix: {confusion_matrix_of_results}")

    print(f"Overall confusion matrix: {EMPTY_MATRIX}")

    for label_index, label in enumerate(labels):

        print(f"Statistics for label {label}:\n")

        print(f"Precision: {precision_array[label_index]}")

        print(f"Recall: {recall_array[label_index]}")

        print(f"F1 score: {f1_score_array[label_index]}\n")