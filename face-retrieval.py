import logging
from pathlib import Path

import streamlit as st
import streamlit_cropper
from PIL import Image
from streamlit_cropper import st_cropper

from utils.image_utils import pil_to_cv2

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="wide")

from face_detection import YOLOv6FaceDetector
from feature_extraction import EdgeFaceFeatureExtractor
from vector_store import MilvusFaceRetriever


@st.cache_resource
def init_face_detector():
    return YOLOv6FaceDetector()


@st.cache_resource
def init_retriever():
    return MilvusFaceRetriever(
        feature_extractor=EdgeFaceFeatureExtractor(), similarity_threshold=0.5
    )


# creating a global default face detector, this must be global
# because the _recommended_box signature is defined
# and does not allow for extra parameters
face_detector = init_face_detector()


def _recommended_bbox_yolo(img: Image, aspect_ratio: tuple, **kwargs) -> dict:
    """Returns the recommended bounding box for the cropper based on the face detector.

    Args:
        img: The image to be processed.
        aspect_ratio: The aspect ratio of the cropper.
    Returns:
        dict: The recommended bounding box. The keys are 'left', 'top', 'width', and 'height'.
    """
    cv2_img = pil_to_cv2(img)
    # the detector returns all the faces in the image, here we assuming only 1 face
    # so we only work with the first element in the list
    detections = face_detector.predict(cv2_img)
    if len(detections) == 0:
        return {"left": 0, "top": 0, "width": int(img.width - 2), "height": (img.height - 2)}

    # converting the bbox to the format expected by the cropper
    left, top, right, bottom = detections[0].bbox

    return {"left": left, "top": top, "width": (right - left), "height": (bottom - top)}


streamlit_cropper._recommended_box = _recommended_bbox_yolo

# logo
st.sidebar.image("assets/sentinel_logo_white.png", use_column_width="always")

# title
st.title("SENTINEL Face Retrieval")

# create a MilvusFaceRetriever object and FeatureExtractor object
retriever = init_retriever()

cols = st.columns(5)

# this returns a file-like object
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    # Get a cropped image from the frontend
    uploaded_img = Image.open(uploaded_file)
    width, height = uploaded_img.size

    new_width = 370
    new_height = int((new_width / width) * height)
    uploaded_img = uploaded_img.resize((new_width, new_height))

    st.sidebar.text(
        "Face to search",
        help="Edit the bounding box to change the ROI (Region of Interest).",
    )
    with st.sidebar.empty():
        cropped_img = st_cropper(
            uploaded_img,
            box_color="#4fc4f9",
            realtime_update=True,
            aspect_ratio=(16, 9),
        )

    show_distance = st.sidebar.toggle("Show confidence", value=True)

    # minimum confidence slider
    value = st.sidebar.slider("Minimum confidence filter", 0.05, 1.0, 0.5, step=0.05)
    retriever.set_similarity_threshold(value)

    # getting hits
    results = retriever.get_search_hits(face_crop=pil_to_cv2(cropped_img))
    # skipping the query image if it is in the db
    results = [result for result in results if result["similarity"] <= 0.99]

    for i, info in enumerate(results):
        imgName = info["filename"]
        score = info["similarity"]
        img = Image.open(imgName)
        cols[i % 5].image(img, use_column_width=True)
        if show_distance:
            cols[i % 5].write(f"Similarity confidence: {score:.3f}")


# This Streamlit app is based on the great bootcamp tutorial from
# the Milvus team: https://github.com/milvus-io/bootcamp/tree/master/bootcamp/tutorials/quickstart/apps/image_search_with_milvus
