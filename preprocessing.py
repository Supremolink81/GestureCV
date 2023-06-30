import torch
from PIL import Image
import torchvision

def preprocess_image(image_name: str) -> torch.Tensor:

    """
    Performs preprocessing on an image.

    In this case, renders the image as a PyTorch tensor, 
    changes its size and scales its side lengths by 12 each.

    Args:

        str image_name: the name of the image to preprocess.

    Returns:

        The preprocessed image as a PyTorch tensor.
    """

    image_object: Image.Image = Image.open(image_name)

    image_object = image_object.convert("RGB")

    image_object = image_object.resize((63, 63), Image.Resampling.LANCZOS)

    image_tensor: torch.Tensor = image_to_tensor(image_object)

    image_object.close()

    del image_object

    return image_tensor

def image_to_tensor(image_object: Image.Image) -> torch.Tensor:

    """
    Converts an image into a PyTorch tensor.

    Args:

        Image.Image image_object: the image object to convert.

    Returns:

        A PyTorch tensor storing the image data.
    """

    image_to_tensor_converter: torchvision.transforms.ToTensor = torchvision.transforms.PILToTensor()

    return image_to_tensor_converter(image_object)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:

    """
    Converts the given tensor to a PIL image.

    Args:

        torch.Tensor tensor: the tensor to convert.

    Returns:

        The tensor converted to a PIL image.
    """

    tensor_to_image_converter: torchvision.transforms.ToPILImage = torchvision.transforms.ToPILImage()

    return tensor_to_image_converter(tensor)