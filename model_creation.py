import torch
from deep_learning_pipelines import ClassificationPipeline
from pipeline_setup import *
import random
import torchvision
import evaluation

if __name__ == "__main__":

    set_random_seed_to_time()

    gpu = torch.device("cuda:0")

    train_set, validation_set, test_set = load_datasets(["processed_data/training_data", "processed_data/validation_data", "processed_data/test_data"], 3 * [True])

    print(f"Test set of size {len(test_set)} loaded")

    loss_function: torch.nn.modules.loss._Loss = torch.nn.CrossEntropyLoss().to(gpu)

    while True:

        set_random_seed_to_time()

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

        label_names: list = ["Middle Finger", "No Gesture", "Ok Sign", "Thumbs Up", "Two Fingers"]

        training_ground_truth, training_model_predictions = pipeline.evaluate(train_set, BATCH_SIZE, gpu)

        validation_ground_truth, validation_model_predictions = pipeline.evaluate(validation_set, BATCH_SIZE, gpu)

        print("Training data results:")

        evaluation.print_evaluation_metrics(training_ground_truth, training_model_predictions, label_names)

        print("Validation data results:")

        evaluation.print_evaluation_metrics(validation_ground_truth, validation_model_predictions, label_names)

        torch.save(pipeline.model.state_dict(), "GestureNN.pth")