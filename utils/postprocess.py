import time
from typing import List, Optional, Tuple

import numpy as np


def scale_coords(
    img1_shape: Tuple[int, int],
    coords: np.ndarray,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[List[float]] = None,
) -> np.ndarray:
    """Rescale coords from img1_shape to img0_shape.

    Args:
        img1_shape: Current shape in HW from where the coords are scaled.
        coords: List of the coords to be scaled. Format (x1y1 ... xNyN) and the length needs to be divisible by 2.
        img0_shape: Shape in HW to which the coords will be scaled.
        ratio_pad (optional): List that stores the gain and pad to be used.

    Returns:
        coords: List of the coords scaled. Format (x1y1 ... xNyN).
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [c * 2 for c in range(len(coords[0]) // 2)]] -= pad[0]  # x padding
    coords[:, [c * 2 + 1 for c in range(len(coords[0]) // 2)]] -= pad[1]  # y padding
    coords[:, : len(coords[0])] /= gain
    clip_coords(coords, img0_shape)

    return coords


def clip_coords(boxes: np.ndarray, shape: Tuple[int, int]):
    """Clip bounding boxes to image shape.

    Args:
        boxes: List of the coords to be cliped. Format (x1y1 ... xNyN) and the length needs to be divisible by 2.
        shape: Shape in HW to which the coords will be cliped.
    """
    boxes[:, [c * 2 for c in range(len(boxes[0]) // 2)]] = boxes[
        :, [c * 2 for c in range(len(boxes[0]) // 2)]
    ].clip(
        0, shape[1]
    )  # x1, x2
    boxes[:, [c * 2 + 1 for c in range(len(boxes[0]) // 2)]] = boxes[
        :, [c * 2 + 1 for c in range(len(boxes[0]) // 2)]
    ].clip(0, shape[0])


def softmax_np(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.20,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: list = (),
    max_det: int = 300,
    extra_data: int = 0,
):
    """Code taken from:
    https://github.com/meituan/YOLOv6/blob/yolov6-face/yolov6/utils/nms.py#L108.

    Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb94
    63c815e9171be955/utils/general.py#L775.

    Args:
        prediction: With shape [N, 5 + extra_data], N is the number of bboxes. The first 4 elements are expected to be bbox of the object.
        conf_thres: Confidence threshold.
        iou_thres: IoU threshold.
        classes: If a list is provided, nms only keep the classes you provide.
        agnostic: When it is set to True, we do class-independent nms, otherwise, different classes would do nms independently of each other.
        multi_label: When it is set to True, one box can have multi labels, otherwise, one box only have one label.
        labels: List of apriori labels to be concatenated if autolabelling.
        max_det: Max number of output bboxes.
        extra_data: Number of extra dimensions in "prediction", that includes additional information.

    Returns:
         list of detections, on (N, 6 + extra_data) tensor per image [xyxy, conf, cls, extra_data]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - (5 + extra_data)  # number of classes

    if isinstance(conf_thres, float):
        xc_thresh = conf_thres  # candidates
    elif isinstance(conf_thres, dict):
        xc_thresh = min(conf_thres.values())
    else:
        raise ValueError(f"Type {type(conf_thres)} not supported for conf_thres")

    assert (
        0 <= xc_thresh <= 1
    ), f"Invalid Confidence threshold {xc_thresh}, valid values are between 0.0 and 1.0"

    xc = prediction[..., 4 + extra_data] > xc_thresh

    # Checks
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs * extra_data  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6 + extra_data))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + 5 + extra_data))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + (5 + extra_data)] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5 + extra_data :] *= x[
            :, 4 + extra_data : 5 + extra_data
        ]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            raise NotImplementedError
            # TODO: better translate to np
            # i, j = np.argwhere(x[:, 5:] > conf_thres).T
            # x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].float()), axis=1)
        else:  # best class only
            j = x[:, 5 + extra_data :].argmax(axis=1, keepdims=True)
            conf = x[:, 5 + extra_data :].max(axis=1, keepdims=True)
            x = np.concatenate(
                (box, conf, j, x[:, 4 : 4 + extra_data]), axis=1
            )  # [xyxy, conf, cls_idx]

            if isinstance(conf_thres, float):
                select = conf.reshape(-1) > conf_thres
            elif isinstance(conf_thres, dict):
                select = np.zeros(x[..., 5 + extra_data].shape, dtype=np.bool)
                for cls, cls_thresh in conf_thres.items():
                    select |= (x[..., 5 + extra_data] == cls) & (
                        x[..., 4 + extra_data] > cls_thresh
                    )

            x = x[select]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(axis=1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # CHECK: not sure if this is equivalent
            # original torch line: x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            x = x[x[:, 4].argsort()[::-1, ...][:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        # TODO: replace
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS cost time exceed the limited {time_limit}s.")
            break  # time limit exceeded

    return output


# from: https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
def nms(boxes, confidence_scores, threshold):
    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return np.array([])

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Picked bounding boxes
    keep_idxs = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(confidence_scores)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        keep_idxs.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(keep_idxs)


# from: https://github.com/ultralytics/yolov5
def xywh2xyxy(x):
    """Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-
    left, x2y2=bottom-right."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y
