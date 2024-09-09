from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt


class FeatureExtractor(ABC):
    """Abstract base class for image feature extractors.

    Defines the common interface for models that extract embeddings from images, ensuring
    consistency in implementation and usage. Concrete feature extractor classes must inherit
    from this class and implement its methods.
    """

    @abstractmethod
    def _init_model(self) -> None:
        """Initializes the detector model, as required by the model itself.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, image: npt.NDArray) -> Any:
        """Performs all preprocessing steps before predicting over an image.

        Args:
            image (npt.NDArray): Image to perform preprocessing on.

        Returns:
            Any: An object representing the image as required by the actual model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: npt.NDArray) -> npt.NDArray:
        """Extracts embeddings from the given image.

        Args:
            image (npt.NDArray): Image to extract features from.

        Returns:
            npt.NDArray: A numpy array containing the extracted embeddings.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
