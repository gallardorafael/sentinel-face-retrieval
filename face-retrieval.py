import logging
from pathlib import Path

import streamlit as st
import streamlit_cropper
from PIL import Image
from streamlit_cropper import st_cropper

from utils.image_utils import pil_to_cv2

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="wide")

from feature_extraction import EdgeFaceFeatureExtractor
from vector_store import MilvusFaceRetriever


def _recommended_box2(img: Image, aspect_ratio: tuple) -> dict:
    width, height = img.size
    return {
        "left": int(0),
        "top": int(0),
        "width": int(width - 2),
        "height": int(height - 2),
    }


streamlit_cropper._recommended_box = _recommended_box2

# logo
st.sidebar.image("assets/sentinel_logo_white.png", use_column_width="always")

# title
st.title("SENTINEL Face Retrieval")

# create a MilvusFaceRetriever object and FeatureExtractor object
retriever = MilvusFaceRetriever(
    feature_extractor=EdgeFaceFeatureExtractor(), similarity_threshold=0.5
)

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

    for i, info in enumerate(results):
        imgName = info["filename"]
        score = info["similarity"]
        img = Image.open(imgName)
        cols[i % 5].image(img, use_column_width=True)
        if show_distance:
            cols[i % 5].write(f"Similarity confidence: {score:.3f}")
