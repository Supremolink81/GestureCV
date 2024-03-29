import torch
import torchvision
from torchvision.transforms.functional import rotate
from PIL import Image, ImageOps, ImageFilter

def create_additional_samples(image_tensor: torch.Tensor) -> list[torch.Tensor]:

    """
    Creates 8 image samples from a single image.

    The first 4 are made via rotations of the original; 0, 90, 
    180 and 270 degrees to the right respectively.

    The next 4 are identical to the first, but with added Gaussian noise.

    Args:

        torch.Tensor image_tensor: the image to create samples from.

    Returns:

        A list of 8 samples created from the original image.
    """

    new_samples: list[torch.Tensor] = []

    image_clone: torch.Tensor = image_tensor.clone()

    for _ in range(4):

        new_samples.append(image_clone)

        new_samples.append(add_noise_to_tensor(image_clone))

        image_clone = rotate(image_clone, 90.0)

    new_samples_expanded: list[torch.Tensor] = []

    for blur_factor in range(4):

        for sample in new_samples:

            sample_as_pil: Image.Image = torchvision.transforms.ToPILImage()(sample)

            blurred_image: Image.Image = sample_as_pil.filter(ImageFilter.BoxBlur(blur_factor))

            blurred_image_mirrored: Image.Image = ImageOps.mirror(blurred_image)

            blurred_image_as_tensor: torch.Tensor = torchvision.transforms.PILToTensor()(blurred_image)

            mirrored_image_as_tensor: Image.Image = torchvision.transforms.PILToTensor()(blurred_image_mirrored)

            new_samples_expanded.append(blurred_image_as_tensor)

            new_samples_expanded.append(mirrored_image_as_tensor)

    return new_samples_expanded

def add_noise_to_tensor(image_tensor: torch.Tensor) -> torch.Tensor:

    """
    Adds Gaussian noise to a given image tensor.

    Gaussian distribution possesses a mean of 0 and
    a standard deviation of 0.05.

    Args:

        Image.Image image_tensor: the image to add noise to.

    Returns:

        the image with added Gaussian noise.
    """

    mean_tensor: torch.Tensor = torch.zeros_like(image_tensor).float()

    std_tensor: torch.Tensor = 0.05 * torch.ones_like(image_tensor).float()

    gaussian_noise: torch.Tensor = torch.normal(mean=mean_tensor, std=std_tensor)

    return image_tensor + gaussian_noise