import torch
from torch.utils import data
import numpy as np
from sklearn.metrics import confusion_matrix

class Pipeline(torch.nn.Module):

    """
    A wrapper class representing a deep learning pipeline.
    
    Allows users to set a model, optimizer and loss function as member
    variables, and train the network using different learning rates,
    epochs, and/or batch sizes.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: torch.nn.modules.loss._Loss

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function: torch.nn.modules.loss._Loss):
        
        super().__init__()

        self.model = model

        self.optimizer = optimizer

        self.loss_function = loss_function

    def train(self, training_data: data.Dataset, epochs: int, batch_size: int = 1, gpu: torch.device = None, learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> list[float]:

        """
        Trains the pipeline's model with the given epochs and batch size.

        At each epoch, runs through a training loop where the architecture is optimized in
        one step of gradient descent according to the optimizer and the batch size.

        After training is complete, returns the list of losses for each
        mini-batch of data at every epoch for potential use in visualzation.

        Args:

            data.Dataset training_data: the data to train the model on.

            int epochs: the number of epochs to train.

            int batch_size: the batch size to use during training.

            torch.device gpu: the GPU device to run training on. If None, runs training on the CPU.

            torch.optim.lr_scheduler._LRScheduler learning_rate_scheduler: the learning rate
            scheduler to use. If None, no scheduler is used.

        Returns:

            None
        """

        training_data_loader: data.DataLoader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)

        loss_values: list[float] = []

        for i in range(epochs):

            for images, labels in training_data_loader:

                images = images.float()

                labels = labels.float()

                if gpu:

                    images = images.to(gpu)

                    labels = labels.to(gpu)

                self.optimizer.zero_grad()

                model_predictions: torch.Tensor = self.model(images)

                loss_result: torch.Tensor = self.loss_function(model_predictions, labels)

                loss_values.append(loss_result.to("cpu").item())

                loss_result.backward()

                self.optimizer.step()

                print(f"Batch {len(loss_values)} done.")

            print(f"Epoch {i+1} done.")

            if learning_rate_scheduler is not None:

                learning_rate_scheduler.step()

        return loss_values
    
    def evaluate(self, test_data: data.Dataset, batch_size: int) -> None:

        assert False # must be implemented
    
class ClassificationPipeline(Pipeline):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function: torch.nn.modules.loss._Loss):

        super().__init__(model, optimizer, loss_function)

    def evaluate(self, test_data: data.Dataset, batch_size: int = 1, gpu: torch.device = None, labels=None) -> float:

        correctly_predicted: int = 0
        test_samples: int = len(test_data)

        test_data_loader: data.DataLoader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        with torch.no_grad():

            ground_truth_labels: list[torch.Tensor] = []

            model_predicted_labels: list[torch.Tensor] = []

            for images, labels in test_data_loader:

                images = images.float()

                labels = labels.float()

                if gpu:

                    images = images.to(gpu)

                    labels = labels.to(gpu)

                model_predictions: torch.Tensor = self.model(images)

                image_labels: torch.Tensor = torch.max(labels, 1)[1]

                class_predictions: torch.Tensor = torch.max(model_predictions, 1)[1]

                ground_truth_labels.append(image_labels.cpu())

                model_predicted_labels.append(class_predictions.cpu())

                correctly_predicted += (class_predictions == image_labels).sum().item()

            ground_truth_array: np.array = torch.cat(ground_truth_labels).numpy()

            model_prediction_array: np.array = torch.cat(model_predicted_labels).numpy()

            evaluation_confusion_matrix: np.array = confusion_matrix(ground_truth_array, model_prediction_array, labels=labels)

        percent_accuracy: float = 100 * correctly_predicted / test_samples

        return percent_accuracy