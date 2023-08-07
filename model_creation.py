import torch
from deep_learning_pipelines import *
from pipeline_setup import *
from evaluation import MetricState
import torchvision
from composer.algorithms.sam import SAMOptimizer

if __name__ == "__main__":

    set_random_seed_to_time()

    gpu = torch.device("cuda:0")

    train_set, validation_set, test_set = load_datasets(["processed_data/training_data", "processed_data/validation_data", "processed_data/test_data"], 3 * [True])

    loss_function: torch.nn.modules.loss._Loss = torch.nn.CrossEntropyLoss().to(gpu)

    iterations: int = int(input("Enter number of iterations to run training: "))

    train_state: MetricState = MetricState({
        "accuracy" : torchmetrics.Accuracy(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "precision" : torchmetrics.Precision(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "recall" : torchmetrics.Recall(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "f1_score" : torchmetrics.F1Score(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "confusion_matrix" : torchmetrics.ConfusionMatrix(task='multiclass', num_classes=5),
    })
    
    validation_state: MetricState = MetricState({
        "accuracy" : torchmetrics.Accuracy(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "precision" : torchmetrics.Precision(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "recall" : torchmetrics.Recall(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "f1_score" : torchmetrics.F1Score(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "confusion_matrix" : torchmetrics.ConfusionMatrix(task='multiclass', num_classes=5),
    })

    for _ in range(iterations):

        set_random_seed_to_time()

        LEARNING_RATE, EPOCHS, BATCH_SIZE, REGULARIZATION_COEFFICIENT = get_training_hyperparameters()

        architecture: torch.nn.Module = torchvision.models.shufflenet_v2_x0_5(num_classes=5).to(gpu)

        optimizer: torch.optim.Optimizer = torch.optim.SGD(architecture.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION_COEFFICIENT)

        sam_optimizer: SAMOptimizer = SAMOptimizer(optimizer)

        learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        print("Architecture, optimizer, loss and scheduler loaded.")

        pipeline: ClassificationPipeline = ClassificationPipeline(architecture, sam_optimizer, loss_function)

        loss_values: list[float] = pipeline.train(train_set, EPOCHS, BATCH_SIZE, gpu, learning_rate_scheduler)

        plot_loss_graph(loss_values)

        print("Model trained, evaluating model...")

        train_ground_truth, train_model_predictions = pipeline.evaluate(train_set, BATCH_SIZE, gpu)

        validation_ground_truth, validation_model_predictions = pipeline.evaluate(validation_set, BATCH_SIZE, gpu)

        train_state.update_state(train_ground_truth, train_model_predictions)

        validation_state.update_state(validation_ground_truth, validation_model_predictions)

        torch.save(pipeline.model.state_dict(), "GestureNN.pth")