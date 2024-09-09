import logging
from typing import List, Optional, TypedDict

import numpy.typing as npt
from pymilvus import MilvusClient

from feature_extraction import EdgeFaceFeatureExtractor, FeatureExtractor

from .defaults import *

logger = logging.getLogger(__name__)


class FaceHit(TypedDict):
    person_name: str
    similarity: float
    filename: str


class MilvusFaceRetriever:
    def __init__(
        self,
        similarity_threshold: Optional[float] = 0.5,
        feature_extractor: Optional[FeatureExtractor] = EdgeFaceFeatureExtractor(),
        db_uri: Optional[str] = MILVUS_URI,
        db_name: Optional[str] = MILVUS_DB_NAME,
        collection_name: Optional[str] = MILVUS_COLLECTION_NAME,
        top_k: Optional[int] = 50,
        output_fields: Optional[List[str]] = DEFAULT_FIELDS,
    ):
        """Initializes the MilvusFaceRetriever object.

        Args:
            similarity_threshold: The minimum similarity score required to consider a hit as a valid hit.
            feature_extractor: A feature extractor object that extracts features from face images.
            db_uri: The URI of the Milvus server.
            db_name: The name of the database in the Milvus server.
            collection_name: The name of the collection in the Milvus server.
            top_k: The number of hits to retrieve from the database, this do not mean that the search will return exactly top_k hits, since
            the actual returned hits depends on the similarity_threshold.
        """
        self.client = MilvusClient(uri=db_uri, db_name=db_name)
        self.feature_extractor = feature_extractor
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.output_fields = output_fields

    def get_search_hits(self, face_crop: npt.NDArray) -> List[FaceHit]:
        """Searches for similar faces in the Milvus collection. Note that while the hits return a
        filename, this filename depends on what was inserted while filling the collection, it may
        be an absolute path or just a filename, depending on the way the data was inserted.

        Args:
            face_crop: A cropped face image.
        Returns:
            A list of FaceHit objects. Each FaceHit object contains the person's name, similarity score, and filename.
        """
        face_vector = self.feature_extractor.predict(face_crop)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[face_vector],
            output_fields=self.output_fields,
            limit=self.top_k,
            params={"metric_type": DEFAULT_METRIC},
        )
        logger.info(
            "%s nn found, filtering by similarity threshold %s",
            len(results[0]),
            self.similarity_threshold,
        )
        hits: List[FaceHit] = []
        for hit in results[0]:
            # Since the DEFAULT_METRIC is COSINE, Milvus is already giving us similarity, not a distance
            if hit["distance"] >= self.similarity_threshold:
                hits.append(
                    {
                        "person_name": hit["entity"].get("name"),
                        "similarity": hit["distance"],
                        "filename": hit["entity"].get("filename"),
                    }
                )
        logger.info("Returning %s hits", len(hits))
        return hits

    def set_similarity_threshold(self, similarity_threshold: float) -> None:
        """Sets the similarity threshold for the retriever.

        Args:
            similarity_threshold: The minimum similarity score required to consider a hit as a valid hit.
        """
        logging.info("Setting similarity threshold to %s", similarity_threshold)
        self.similarity_threshold = similarity_threshold

    def __del__(self):
        """Closes the Milvus client when the object is destroyed."""
        self.client.close()
