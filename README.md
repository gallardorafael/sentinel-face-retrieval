# SENTINEL Face Retrieval

SENTINEL Face Retrieval is an image similarity search system that allows for extremely fast face retrieval, even when the search space has 1+ million points. Designed to work with CPU-only machines, focused on edge applications where the compute resources are limited.

Note: While SENTINEL Face Retrieval is optimized to work on CPU-only machines, and it would be certainly faster with GPUs, the application is extremely fast, check the Performance section.

### Architecture

This repository has three main modules:

- Embedding (feature_extraction): An ultra-lightweight face feature (embeddings) extractor, based on [EdgeFace](https://github.com/otroshi/edgeface) - a model optimized to work with edge devices. The default model here is the small version, with 3.65 million parameters and around 3 Gflops. This implementation is aimed to work on CPU-only machines (onnxruntime), but it can be accelerated with GPUs also (onnxruntime-gpu).
- Store (vector_store): A vector store based on the great [Milvus](https://github.com/milvus-io/milvus) vector database. This is the most important part of the whole face retrieval system, it is here where all the embeddings are stored, and it is Milvus the one that powers the super-fast similarity search features over which our FaceRetrieval object is created.
- FaceRetriever (vector_store): A wrapper of the MilvusClient.search function, that removes most of the boilerplate by using several default values, the user can still configure everything, but this helps to reduce the code one needs to write to perform search operations in the vector database. By default, this retriever gets the top-50 most similar images to the query, and then applies a threshold (default to 0.5 but customizable) to filter only the images with a similarity above that threshold.

<p align="center">
  <img src="assets/readme/face-retrieval-architecture.png" align="middle" width = "1000" />
</p>

Additionally, some helper functionalities are included:

- Face detector (face_detection): A lightweight face detector based on [YOLOv6-face](https://github.com/meituan/YOLOv6/tree/yolov6-face), the default model is the nano version of the detector, with only 4.63 million parameters, and 11.35 Gflops. Implementing bigger and better versions of this detector is straightforward. This implementation is aimed to work on CPU-only machines (onnxruntime), but it can be accelerated with GPUs also (onnxruntime-gpu). This is useful to detect faces while inserting data into Milvus (EdgeFace only works with face crops) and while recommending a bbox for the query in the Streamlit app.

## Set up

## Usage

## Performance notes

## Acknowledgments
