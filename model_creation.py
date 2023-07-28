import torch
from deep_learning_pipelines import ClassificationPipeline
from pipeline_setup import *
import random
import time
import os
import torchvision

if __name__ == "__main__":

    random_seed: int = time.time()

    print(random_seed)

    random.seed(random_seed)

    gpu = torch.device("cuda:0")

    train_set: GestureDataset = load_dataset("processed_data/training_data", preload_tensors=True)

    print(f"Training set of size {len(train_set)} loaded.")

    validation_set: GestureDataset = load_dataset("processed_data/validation_data", preload_tensors=True)

    print(f"Validation set of size {len(validation_set)} loaded.")

    test_set: GestureDataset = load_dataset("processed_data/test_data", preload_tensors=True)

    print(f"Test set of size {len(test_set)} loaded")

    loss_function: torch.nn.modules.loss._Loss = torch.nn.CrossEntropyLoss().to(gpu)

    while True:

        LEARNING_RATE, EPOCHS, BATCH_SIZE, REGULARIZATION_COEFFICIENT = get_training_hyperparameters()

        architecture: torch.nn.Module = torchvision.models.shufflenet_v2_x0_5(num_classes=5)

        optimizer: torch.optim.Optimizer = torch.optim.SGD(architecture.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION_COEFFICIENT)

        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        print("Architecture, optimizer, loss and scheduler loaded.")

        pipeline: ClassificationPipeline = ClassificationPipeline(architecture, optimizer, loss_function).to(gpu)

        loss_values: list[float] = pipeline.train(train_set, EPOCHS, BATCH_SIZE, gpu, learning_rate_scheduler)

        print(len(loss_values))

        plot_loss_graph(loss_values)

        print("Model trained, evaluating model...")

        training_accuracy: float = pipeline.evaluate(train_set, BATCH_SIZE, gpu, [0, 1, 2, 3, 4])

        validation_accuracy: float = pipeline.evaluate(validation_set, BATCH_SIZE, gpu, [0, 1, 2, 3, 4])

        print(f"{training_accuracy}% training accuracy and {validation_accuracy}% validation accuracy achieved.")

        torch.save(pipeline.model.state_dict(), "GestureNN.pth")