import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils import data
import os
import random
import time
import preprocessing
import data_expansion
    
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
    
def preprocess_image_data(
        raw_image_data_folder: str, 
        training_image_data_folder: str, 
        validation_image_data_folder: str,
        test_image_data_folder: str,
        label_split_amounts: tuple[int, int, int],
    ) -> None:

    """
    Preprocesses data in a raw data folder and stores it in another folder.

    This assumes `raw_image_data_folder` contains directories corresponding to 
    label folders, and that `processed_image_data_folder` also contains said directories.

    Args:

        `str raw_image_data_folder`: the folder containing the raw data.

        `str training_image_data_folder`: the folder to place the training data in.

        `str validation_image_data_folder`: the folder to place the validation data in.

        `str test_image_data_folder`: the folder to place the test data in.

        `tuple[int, int, int] label_split_amounts`: the split amounts for the training, validation
        and test sets for each label.

    Returns:

        None
    """

    training_image_file_paths: list[str] = []

    validation_image_file_paths: list[str] = []

    test_image_file_paths: list[str] = []

    for directory in os.listdir(raw_image_data_folder):

        print(directory)

        labelled_image_file_names: list[str] = list(os.listdir(raw_image_data_folder + "/" + directory))

        random.shuffle(labelled_image_file_names)

        convert_image_name_to_path: callable = lambda image_name: raw_image_data_folder + "/" + directory + "/" + image_name

        labelled_image_file_paths: list[str] = list(map(convert_image_name_to_path, labelled_image_file_names))

        labelled_training_image_file_paths: list[str] = labelled_image_file_paths[ : label_split_amounts[0]]

        labelled_validation_image_file_paths: list[str] = labelled_image_file_paths[label_split_amounts[0] : label_split_amounts[0] + label_split_amounts[1]]

        labelled_test_image_file_paths: list[str] = labelled_image_file_paths[label_split_amounts[0] + label_split_amounts[1] : ]

        training_image_file_paths += labelled_training_image_file_paths

        validation_image_file_paths += labelled_validation_image_file_paths

        test_image_file_paths += labelled_test_image_file_paths

    load_image_file_paths(training_image_file_paths, training_image_data_folder, True)

    load_image_file_paths(validation_image_file_paths, validation_image_data_folder)

    load_image_file_paths(test_image_file_paths, test_image_data_folder)

def load_datasets(paths: list[str], preload_tensors: list[bool]) -> list[GestureDataset]:

    """
    Loads a list of paths represenging

    Args:

        `str paths`: the paths to the folders to load.

        `bool preload_tensors`: whether to preload the tensor data for each folder.
    """

    gesture_datasets: list[GestureDataset] = []

    for path, preload_tensors_for_path in zip(paths, preload_tensors):

        loaded_dataset: GestureDataset = load_dataset(path, preload_tensors_for_path)
    
        gesture_datasets.append(loaded_dataset)

        print(f"Data from path {path} loaded.")

    return gesture_datasets

def load_image_file_paths(image_file_paths: list[str], folder_path: str, augument: bool = False) -> None:

    """
    Loads a collection of image file paths into a specified folder. Images are assumed to 
    be in PNG format.

    Args:

        `list[str] image_file_paths`: the paths to the images to load.

        `str folder_path`: the folder to place the preprocessed images in.

        `bool augument`: whether to augment the data through transformations. Default: False

    Returns:   

        None
    """

    for image_file_path in image_file_paths:

        image_tensor: torch.Tensor = preprocessing.preprocess_image(image_file_path)

        changed_image_path: str = image_file_path[image_file_path.find("/")+1:]

        if augument:

            image_tensors: list[torch.Tensor] = data_expansion.create_additional_samples(image_tensor)

            for image_index, augmented_image_tensor in enumerate(image_tensors):

                image_object: Image.Image = preprocessing.tensor_to_image(augmented_image_tensor)

                image_object.save(folder_path + "/" + changed_image_path[:-4] + f"{image_index}.png")
        
        else:

            image_object: Image.Image = preprocessing.tensor_to_image(image_tensor)

            image_object.save(folder_path + "/" + changed_image_path)

def load_dataset(path: str, preload_tensors: bool = False) -> GestureDataset:

    """
    Loads image data from a given path. The path is expected to be to a folder
    that contains the directories ['middle_finger', 'no_gesture', 'ok_sign', 'thumbs_up', 'two_fingers'],
    which themselves contain image data.

    Args:

        `str path`: the path to the folder to load.

        `bool preload_tensors`: whether to preload the tensor data.

    Returns:

        A GestureDataset object containing the image paths.
    """

    image_paths: list[str] = []

    for image_label_folder in os.listdir(path):

        get_full_image_path: callable = lambda image_name: path + "/" + image_label_folder + "/" + image_name

        images_in_label_folder: list[str] = list(map(get_full_image_path, os.listdir(path + "/" + image_label_folder)))

        image_paths += images_in_label_folder

    return GestureDataset(image_paths, preload_tensors=preload_tensors)

def get_training_hyperparameters() -> tuple[float, int, int, float]:

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

        `list[float] loss_values`: the list of loss values to graph.

    Returns:

        None
    """

    plt.plot(loss_values)

    plt.xlabel("Batch Number")

    plt.ylabel("Loss Value")

    plt.show()

def set_random_seed_to_time() -> None:

    """
    Sets the random seed to the current system time.

    Args:

        None

    Returns:

        None
    """

    random_seed: int = time.time()

    print(f"Random seed: {random_seed}")

    random.seed(random_seed)