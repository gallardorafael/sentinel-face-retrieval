from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from .feature_extractor import FeatureExtractor

DEFAULT_MODELS_PATH = Path(__file__).parent / "models"


class EdgeFaceFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        input_shape: Optional[Tuple[int, int]] = (112, 112),
        model_path: Optional[Path] = DEFAULT_MODELS_PATH / "edgeface_s_gamma_05.onnx",
    ) -> None:
        """Initializes a EdgeFaceFeatureExtractor object from a provided path to an ONNX file. This
        class also includes all the preprocessing steps required by the EdgeFace model.

        Model reference: https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface

        Args:
            input_shape (Optional[Tuple[int, int]]): Shape of the input image. Defaults to (112, 112).
            model_path (Optional[Path]): Path to the ONNX model file. Defaults to the EdgeFace model.
        """
        self.input_shape = input_shape
        self._init_model(model_path=model_path)

    def _init_model(self, model_path: str) -> None:
        """Initializes the ONNX model by loading it from the provided model path.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        self.ort_session = ort.InferenceSession(model_path)

    def _preprocess(self, image: npt.NDArray) -> npt.NDArray:
        """Performs all preprocessing steps before predicting over an image. The EdgeFace model
        requires an input of size 112x112, with torch style norm,

        (img / 255. - 0.5) / 0.5, and with order BxCxWxH

        Args:
            image (npt.NDArray): Image to perform preprocessing on, shape (H, W, C) is expected.

        Returns:
            npt.NDArray: An object representing the image as required by the actual model, with shape
            (1, C, H, W).
        """
        # resize image
        preprocessed_image = cv2.resize(image, self.input_shape).astype(np.float32)
        # 0 - 255 to 0.0 - 1.0 with mean and std of 0.5
        preprocessed_image = (preprocessed_image / 255.0 - 0.5) / 0.5
        # HWC to CHW
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))
        # adding batch dim
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        return preprocessed_image

    def predict(self, image: npt.NDArray) -> npt.NDArray:
        """Predicts the embeddings for the given image. Note that the image is expected to be a
        crop of a face, since this model is not responsible for face detection.

        Args:
            image (npt.NDArray): The input image, with shape (H, W, C).

        Returns:
            npt.NDArray: The embeddings of the image, with shape (512,).
        """
        preprocessed_image = self._preprocess(image)
        embeddings = self.ort_session.run(None, {"data": preprocessed_image})[0]

        # removing batch dim
        embeddings = embeddings.squeeze()

        return embeddings
