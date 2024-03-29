import torch
from deep_learning_pipelines import *
from pipeline_setup import *
from evaluation import MetricState
from composer.algorithms.sam import SAMOptimizer
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay

if __name__ == "__main__":

    train_state: MetricState = MetricState({
        "Accuracy" : torchmetrics.Accuracy(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "Precision" : torchmetrics.Precision(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "Recall" : torchmetrics.Recall(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "F1 Score" : torchmetrics.F1Score(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "Confusion Matrix" : torchmetrics.ConfusionMatrix(task='multiclass', num_classes=5),
    })

    print("Training state initialized.")
    
    validation_state: MetricState = MetricState({
        "Accuracy" : torchmetrics.Accuracy(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "Precision" : torchmetrics.Precision(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "Recall" : torchmetrics.Recall(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "F1 Score" : torchmetrics.F1Score(task='multiclass', average='none', num_classes=5, multidim_average='samplewise'),
        "Confusion Matrix" : torchmetrics.ConfusionMatrix(task='multiclass', num_classes=5),
    })

    print("Validation state initialized.")

    set_random_seed_to_time()

    gpu = torch.device("cuda:0")

    LABEL_LIST: list[str] = ["Middle Finger", "No Gesture", "Ok Sign", "Thumbs Up", "Two Fingers"]

    train_set, validation_set, test_set = load_datasets(["processed_data/training_data", "processed_data/validation_data", "processed_data/test_data"], 3 * [True])

    loss_function: torch.nn.modules.loss._Loss = torch.nn.CrossEntropyLoss().to(gpu)

    iterations: int = int(input("Enter number of iterations to run training: "))

    for _ in range(iterations):

        set_random_seed_to_time()

        LEARNING_RATE, EPOCHS, BATCH_SIZE, REGULARIZATION_COEFFICIENT, STEP_SIZE = get_training_hyperparameters()

        architecture: torch.nn.Module = torchvision.models.shufflenet_v2_x0_5(num_classes=5).to(gpu)

        optimizer: torch.optim.Optimizer = torch.optim.SGD(architecture.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION_COEFFICIENT)

        sam_optimizer: SAMOptimizer = SAMOptimizer(optimizer)

        learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

        print("Architecture, optimizer, loss and scheduler loaded.")

        pipeline: ClassificationPipeline = ClassificationPipeline(architecture, optimizer, loss_function)

        loss_values: list[float] = pipeline.train(train_set, EPOCHS, BATCH_SIZE, gpu, learning_rate_scheduler)

        plot_loss_graph(loss_values)

        print("Model trained, evaluating model...")

        train_ground_truth, train_model_predictions = pipeline.evaluate(train_set, BATCH_SIZE, gpu)

        validation_ground_truth, validation_model_predictions = pipeline.evaluate(validation_set, BATCH_SIZE, gpu)

        print("Evaluation data gathered.")

        train_state.update_state(train_ground_truth.reshape((1,) + train_ground_truth.shape), train_model_predictions.reshape((1,) + train_model_predictions.shape))

        print("Training state updated.")

        print(train_state)

        validation_state.update_state(validation_ground_truth.reshape((1,) + validation_ground_truth.shape), validation_model_predictions.reshape((1,) + validation_model_predictions.shape))

        print("Validation state updated.")

        print(validation_state)

        train_confusion_matrix_display: ConfusionMatrixDisplay = ConfusionMatrixDisplay(train_state.state["Confusion Matrix"].compute().numpy(), display_labels=np.array(LABEL_LIST))

        train_confusion_matrix_display.plot()

        plt.show()

        validation_confusion_matrix_display: ConfusionMatrixDisplay = ConfusionMatrixDisplay(validation_state.state["Confusion Matrix"].compute().numpy(), display_labels=np.array(LABEL_LIST))

        validation_confusion_matrix_display.plot()

        plt.show()

        torch.save(pipeline.model.state_dict(), "GestureNN.pth")