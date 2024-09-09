import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(image: Image) -> np.ndarray:
    """
    Converts a PIL image to a numpy array in the OpenCV format.
    Args:
        image: A PIL image.
    Returns:
        A numpy array in the OpenCV format.

    Borrowed from: https://gist.github.com/panzi/1ceac1cb30bb6b3450aa5227c02eedd3
    """
    mode = image.mode
    new_image: np.ndarray
    if mode == "1":
        new_image = np.array(image, dtype=np.uint8)
        new_image *= 255
    elif mode == "L":
        new_image = np.array(image, dtype=np.uint8)
    elif mode == "LA" or mode == "La":
        new_image = np.array(image.convert("RGBA"), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == "RGB":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == "RGBA":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == "LAB":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_LAB2BGR)
    elif mode == "HSV":
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
    elif mode == "YCbCr":
        # XXX: not sure if YCbCr == YCrCb
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_YCrCb2BGR)
    elif mode == "P" or mode == "CMYK":
        new_image = np.array(image.convert("RGB"), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == "PA" or mode == "Pa":
        new_image = np.array(image.convert("RGBA"), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"unhandled image color mode: {mode}")

    return new_image
