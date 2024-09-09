from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class Detection:
    """Represents a detected object using a dataclass for conciseness and immutability.

    Attributes:
        class_id (str): Class name of the detected object.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2) of the detected object.
        confidence (float): Confidence score of the detection.
        face_landmarks (np.ndarray): Face landmark if available.
    """

    detection_type: str
    bbox: tuple
    confidence: float
    face_landmarks: Optional[np.ndarray] = None


class Detector(ABC):
    """Abstract base class for object detectors.

    Defines the common interface for object detectors, ensuring consistency in
    implementation and usage. Concrete detector classes must inherit from this class and
    implement its methods.
    """

    @abstractmethod
    def _init_model(self) -> None:
        """Initializes the detector model, as required by the model itself.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Any) -> List[Detection]:
        """Detects objects in the given image.

        Args:
            image (Any): Image to perform object detection on. The preprocessing funtion
                will be in charge of doing all preprocessing.

        Returns:
            List[Detection]: A list of detected objects, where each object
                is represented as a Detection object.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
