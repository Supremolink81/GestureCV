import torch
from matplotlib import pyplot as plt
from torch.utils import data
import os
import preprocessing
    
class GestureDataset(data.Dataset):

    image_names: list[str]
    image_tensors: list[torch.Tensor]

    def __init__(self, image_names, preload_tensors: bool = False):

        self.image_names = image_names

        self.image_tensors = None

        if preload_tensors:

            self.image_tensors = list(map(preprocessing.preprocess_image, image_names))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        image_name: str = self.image_names[index]

        if self.image_tensors is not None:

            image_tensor: torch.Tensor= self.image_tensors[index]

        else:

            image_tensor: torch.Tensor = preprocessing.preprocess_image(image_name)

        image_label_tensor: torch.Tensor = torch.Tensor([1, 0, 0, 0, 0])

        if "middle_finger" in image_name:

            image_label_tensor = torch.tensor([1, 0, 0, 0, 0]).float()

        if "no_gesture" in image_name:

            image_label_tensor = torch.tensor([0, 1, 0, 0, 0]).float()

        if "ok_sign" in image_name:

            image_label_tensor = torch.tensor([0, 0, 1, 0, 0]).float()

        if "thumbs_up" in image_name:

            image_label_tensor = torch.tensor([0, 0, 0, 1, 0]).float()

        if "two_fingers" in image_name:

            image_label_tensor = torch.tensor([0, 0, 0, 0, 1]).float()

        return (image_tensor, image_label_tensor)
    
    def __len__(self):

        return len(self.image_names)

def load_dataset(path: str, preload_tensors: bool = False) -> GestureDataset:

    """
    Loads image data from a given path. The path is expected to be to a folder
    that contains the directories ['middle_finger', 'no_gesture', 'ok_sign', 'thumbs_up', 'two_fingers'],
    which themselves contain image data.

    Args:

        str path: the path to the folder to load.

        bool preload_tensors: whether to preload the tensor data.

    Returns:

        A GestureDataset object containing the image paths.
    """

    image_paths: list[str] = []

    for image_label_folder in os.listdir(path):

        get_full_image_path: callable = lambda image_name: path + "/" + image_label_folder + "/" + image_name

        images_in_label_folder: list[str] = list(map(get_full_image_path, os.listdir(path + "/" + image_label_folder)))

        image_paths += images_in_label_folder

    return GestureDataset(image_paths, preload_tensors=preload_tensors)

def get_training_hyperparameters() -> tuple[float, int, int]:

    """
    Helper function to get the learning rate, epochs, batch size and L2 regularization coefficient.

    Args:

        None

    Returns:

        A tuple containing the learning rate, epochs, batch size and L2 regularization coefficient.
    """

    LEARNING_RATE: float = float(input("Enter learning rate: "))

    EPOCHS: int = int(input("Enter epochs: "))

    BATCH_SIZE: int = int(input("Enter batch size: "))

    REGULARIZATION_COEFFICIENT: float = float(input("Enter regularization coefficient: "))

    return (LEARNING_RATE, EPOCHS, BATCH_SIZE, REGULARIZATION_COEFFICIENT)

def plot_loss_graph(loss_values: list[float]) -> None:

    """
    Displays a loss graph from a list of loss function values.

    Args:

        list[float] loss_values: the list of loss values to graph.

    Returns:

        None
    """

    plt.plot(loss_values)

    plt.xlabel("Batch Number")

    plt.ylabel("Loss Value")

    plt.show()