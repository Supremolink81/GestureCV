import cv2
import time
from PIL import Image
import torch
import torchvision
import numpy as np
from preprocessing import *

def prepare_image_for_inference(image_frame: np.array) -> torch.Tensor:

    """
    Prepares a NumPy array representing a camera frame for inference
    using an image classification neural network.

    Args:

        np.array image_frame: the frame to prepare.

    Returns:

        the image frame ready for inference.
    """

    image_object: Image.Image = Image.fromarray(image_frame).convert("RGB")

    image_object = image_object.resize((63, 63))

    pil_to_tensor_transform: torchvision.transforms.PILToTensor = torchvision.transforms.PILToTensor()

    image_tensor: torch.Tensor = pil_to_tensor_transform(image_object)

    return image_tensor.reshape((1, image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]))

def display_image_label(label_tensor: torch.Tensor) -> str:

    """
    Returns the label gesture of the image.

    Args:

        torch.Tensor label_tensor: the tensor to label.

    Returns:

        one of:

            - Middle Finger
            - No Gesture
            - Ok Sign
            - Thumbs Up
            - Two Fingers
    """

    gesture_options: list[str] = ["Middle Finger", "No Gesture", "Ok Sign", "Thumbs Up", "Two Fingers"]

    print(label_tensor)

    return gesture_options[torch.argmax(label_tensor, dim=-1).item()]

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)

    camera_running: bool = True

    while camera_running:

        ret, frame = camera.read()

        if ret:

            image_tensor: torch.Tensor = (prepare_image_for_inference(frame) / 255.0).to(torch.device("cuda:0")).float()

            gesture_model: torch.nn.Module = torchvision.models.shufflenet_v2_x0_5(num_classes=5).to(torch.device("cuda:0"))

            gesture_model.load_state_dict(torch.load("GestureNN_current.pth"))

            image_label: torch.Tensor = gesture_model(image_tensor)

            cv2.imshow(f"Neural Net Test", frame)

            print(display_image_label(image_label))

            time.sleep(3)

        if cv2.waitKey(1) and 0xFF == ord('q'):

            camera_running = False

    camera.release()

    cv2.destroyAllWindows()