import torchmetrics
import torch

class MetricState:

    state: dict[str, torchmetrics.Metric]

    def __init__(self, state: dict[str, torchmetrics.Metric]):

        self.state = state

    def update_state(self, ground_truth: torch.Tensor, model_predictions: torch.Tensor) -> None:

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

            state_string += f"  {state_name}: "

            try:

                state_string += f"{state_value.compute()}\n"

            except ValueError:

                state_string += f"{torch.zeros(state_value.num_classes, dtype=torch.float16)}\n"

        return f"MetricState(\n{state_string})"