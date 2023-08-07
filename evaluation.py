import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torchmetrics

class MetricState:

    state: dict[str, torchmetrics.Metric]

    def __init__(self, state: dict[str, torchmetrics.Metric]):

        self.state = state

    def update_state(self, ground_truth: np.array, model_predictions: np.array) -> None:

        """
        Updates the state history and running average state of a metric
        with the given ground truth and model predictions.

        Args:

            `str state_name`: the name of the metric state to change.
        """

        for state_value in self.state.values():

            state_value.update(model_predictions, ground_truth)

    def __repr__(self):

        state_string: str = ""

        for state_name, state_value in self.state.items():

            state_string += state_name + 

        return f"MetricState(\n{})"