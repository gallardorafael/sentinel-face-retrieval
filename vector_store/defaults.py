import os

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "default")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "test_collection")
EDGE_FACE_DIM = 512
DEFAULT_VECTOR_FIELD_NAME = "embedding"
DEFAULT_METRIC = "COSINE"
DEFAULT_FIELDS = ["filename", "name"]
