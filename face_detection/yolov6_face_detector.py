from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from utils import letterbox_yolov6, non_max_suppression, scale_coords

from .detector import Detection, Detector

DEFAULT_MODELS_PATH = Path(__file__).parent / "models"


class YOLOv6FaceDetector(Detector):
    def __init__(
        self,
        input_shape: Optional[Tuple[int, int]] = (640, 640),
        model_path: Optional[Path] = DEFAULT_MODELS_PATH / "yolov6n_face.onnx",
    ) -> None:
        """Initializes a YOLOv6FaceDetector object from a provided path to an ONNX file. This class
        also includes all the preprocessing steps required by the YOLOv6_face model.

        Model reference: https://github.com/meituan/YOLOv6/tree/yolov6-face

        Designed to work with models with no built-in NMS plugin.
        Each detection has the following structure in its result:
            result[0:4]: Bounding box (x1, y1, x2, y2)
            result[4]: Confidence of the detection
            result[5]: Object detected
            result[6:]: Coords of face landmarks (left_eye_cx, left_eye_cy,
                                                right_eye_cx, right_eye_cy,
                                                nose_cx, nose_cy,
                                                lip_x1, lip_y1,
                                                lip_x2, lip_y2)

        Args:
            input_shape (Optional[Tuple[int, int]]): Shape of the input image. Defaults to (640, 640).
            model_path (Optional[Path]): Path to the ONNX model file. Defaults to the YOLOv6_face model.
        """
        self.input_shape = input_shape
        self._init_model(model_path)

    def _init_model(self, model_path: str) -> None:
        """Initializes the ONNX model by loading it from the provided model path. The model is
        initialized as an onnxruntime InferenceSession object.

        Args:
            model_path (str): Path to the ONNX model file.
        """
        self.ort_session = ort.InferenceSession(model_path)

    def _preprocess(self, image: npt.NDArray) -> npt.NDArray:
        """Performs all preprocessing steps before predicting over an image. Preprocessing done
        according to:
        https://github.com/meituan/YOLOv6/blob/yolov6-face/yolov6/core/inferer.py#L180.

        Args:
            image: Image to perform preprocessing on. This assumes an input with cv2 format
                HWC and will be converted to CHW, also assumed BGR to be converted to RGB

        Returns:
            preprocessed_image: An object representing the image as required by the actual model.
        """
        # letterboxing square image to 640, 640
        letterboxed_image = letterbox_yolov6(image, new_shape=self.input_shape, auto=False)[0]

        # HWC to CHW, BGR to RGB
        preprocessed_image = letterboxed_image.transpose((2, 0, 1))[::-1].astype(np.float32)

        # 0 - 255 to 0.0 - 1.0
        preprocessed_image /= 255

        # adding batch dim
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

        # making the array contiguous
        preprocessed_image = np.ascontiguousarray(preprocessed_image)

        return preprocessed_image

    def _postprocess(
        self,
        detections,
        original_img_shape: Tuple[int, int],
        confidence_thresh: float = 0.3,
    ):
        nms_results = non_max_suppression(
            detections,
            conf_thres=confidence_thresh,
            agnostic=False,
            extra_data=10,
            max_det=100,
        )[0]

        # # reescaling face bboxes
        nms_results[:, :4] = scale_coords(self.input_shape, nms_results[:, :4], original_img_shape)

        # # reescaling face landmarks
        nms_results[:, 6:] = scale_coords(self.input_shape, nms_results[:, 6:], original_img_shape)

        formatted_detections = []
        for det in nms_results:
            if len(det):
                formatted_detections.append(
                    {"bbox": det[:4], "confidence": det[4], "class": det[5], "landmarks": det[6:]}
                )

        return formatted_detections

    def predict(self, image: npt.NDArray) -> Detection:
        """Detects objects in the given image.

        Args:
            image (npt.NDArray): Image to perform object detection on. The preprocessing funtion
                will be in charge of doing all preprocessing.

        Returns:
            List[Detection]: A list of detected objects, where each object
                is represented as a Detection object. The bboxes are in format (x1, y1, x2, y2).
                An empty list will be returned if no faces were found.
        """
        preprocessed_image = self._preprocess(image)

        detections = self.ort_session.run(None, {"images": preprocessed_image})[0]

        postprocessed_dets = self._postprocess(detections, image.shape[:2])

        detections = []
        for face in postprocessed_dets:
            detection = Detection(
                detection_type="FACE",
                bbox=[int(coord) for coord in face["bbox"]],
                confidence=face["confidence"],
                face_landmarks=face["landmarks"],
            )
            detections.append(detection)

        return detections
