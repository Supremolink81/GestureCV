from typing import Any, Dict, Optional
import torch
from torch.utils import data
import numpy as np
from composer.models import ComposerModel
from composer.trainer.trainer import Trainer
from composer.algorithms import ChannelsLast, LabelSmoothing, BlurPool, SAM, MixUp
import torchmetrics
from torchmetrics import Metric

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

    def evaluate(self, test_data: data.Dataset, batch_size: int = 1, gpu: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:

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

            ground_truth_array: np.array = torch.cat(ground_truth_labels).numpy()

            model_prediction_array: np.array = torch.cat(model_predicted_labels).numpy()

        return ground_truth_array, model_prediction_array
    
class ComposerClassificationModelWrapper(ComposerModel):

    """
    A custom wrapper class to enable support for MosaicML's composer module.
    """

    model: torch.nn.Module
    loss_function: torch.nn.modules.loss._Loss
    num_classes: int

    def __init__(self, model: torch.nn.Module, loss_function: torch.nn.modules.loss._Loss, num_classes: int = 2):

        self.model = model

        self.loss_function = loss_function

        self.num_classes = num_classes

    # overriden ComposerModel method
    def forward(self, input_tensor: torch.Tensor):

        return self.model(input_tensor)
    
    # overriden ComposerModel method
    def loss(self, outputs: torch.Tensor, batch: torch.Tensor, *args, **kwargs):

        return self.loss_function(outputs, batch)
    
    # overriden ComposerModel method
    def eval_forward(self, batch: Any, _: Any | None = None) -> Any:

        return self.forward(batch)
    
    # overriden ComposerModel method
    def get_metrics(self, is_train: bool) -> Dict[str, Metric]:

        return {
            "accuracy" : torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=self.num_classes),
            "precision" : torchmetrics.Precision(task='multiclass', average='micro', num_classes=self.num_classes),
            "recall" : torchmetrics.Recall(task='multiclass', average='micro', num_classes=self.num_classes),
            "f1_score" : torchmetrics.F1Score(task='multiclass', average='micro', num_classes=self.num_classes),
            "confusion_matrix" : torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes),
        }
    
class ImprovedClassificationPipeline:

    """
    The classification pipeline I made above, but improved with the
    optimizations present in MosaicML.
    """

    model: ComposerClassificationModelWrapper
    optimizer: torch.optim.Optimizer

    def __init__(self, model: ComposerClassificationModelWrapper, optimizer: torch.optim.Optimizer):

        self.model = model

        self.optimizer = optimizer

    def train(self, dataset: data.Dataset, epochs: int, batch_size: int = 1, gpu: torch.device = None, learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler = None):

        self.model.train()

        training_dataloader: data.DataLoader = data.DataLoader(dataset, batch_size, shuffle=True)
        
        model_trainer: Trainer = self._make_model_trainer(training_dataloader, learning_rate_scheduler)

        model_trainer.fit()

    def eval(self, dataset: data.Dataset, batch_size: int = 1, gpu: torch.device = None):

        self.model.eval()

        evaluation_dataloader: data.DataLoader = data.DataLoader(dataset, batch_size, shuffle=False)

        model_evaluator: Trainer = self._make_model_trainer(evaluation_dataloader, None)

        model_evaluator.eval()

    def _make_model_trainer(self, training_dataloader: data.DataLoader, learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler):

        channels_last: ChannelsLast = ChannelsLast()

        label_smoothing: LabelSmoothing = LabelSmoothing(smoothing=0.1)

        blur_pool: BlurPool = BlurPool(blur_first=True, replace_maxpools=True, replace_convs=True)

        sharpness_aware_minimization: SAM = SAM(rho=0.05)

        mixup: MixUp = MixUp(alpha=1.0)

        algorithms_used: list = [
            channels_last,
            label_smoothing,
            blur_pool,
            sharpness_aware_minimization,
            mixup,
        ]

        return Trainer(
            model=self.model,
            train_dataloader=training_dataloader,
            optimizers=self.optimizer,
            schedulers=learning_rate_scheduler,
            algorithms=algorithms_used,
        )