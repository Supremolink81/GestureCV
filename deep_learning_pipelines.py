from typing import Any, Dict, Optional
import torch
from torch.utils import data
import numpy as np
import torchmetrics
from torchmetrics import Metric
import composer.functional as cf
from composer.algorithms.sam import SAMOptimizer

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

        cf.apply_blurpool(self.model)

        self.optimizer = optimizer

        self.loss_function = loss_function

    def train(self, training_data: data.Dataset, epochs: int, batch_size: int = 1, gpu: torch.device = None, learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler = None) -> list[float]:

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

                #images, labels, _ = cf.mixup_batch(images, labels, alpha=1.0)

                self.optimizer.zero_grad()

                model_predictions: torch.Tensor = self.model(images)

                cf.smooth_labels(model_predictions, labels, smoothing=0.1)

                loss_result: torch.Tensor = self.loss_function(model_predictions, labels)

                loss_values.append(loss_result.to("cpu").item())

                loss_result.backward()

                if isinstance(self.optimizer, SAMOptimizer):

                    self.optimizer.first_step()

                    self.optimizer.zero_grad()

                    second_loss_result: torch.Tensor = self.loss_function(self.model(images), labels)

                    second_loss_result.backward()

                    self.optimizer.second_step()

                else:

                    self.optimizer.step()

                print(f"Batch {len(loss_values)} done.")

            print(f"Epoch {i+1} done.")

            if learning_rate_scheduler is not None:

                learning_rate_scheduler.step()

        return loss_values
    
class ClassificationPipeline(Pipeline):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_function: torch.nn.modules.loss._Loss):

        super().__init__(model, optimizer, loss_function)

    def evaluate(self, test_data: data.Dataset, batch_size: int = 1, gpu: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:

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

            ground_truth_array: torch.Tensor = torch.cat(ground_truth_labels)

            model_prediction_array: torch.Tensor = torch.cat(model_predicted_labels)

        return ground_truth_array, model_prediction_array