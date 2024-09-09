import argparse
import logging
import pprint
from pathlib import Path

import cv2
from pymilvus import MilvusClient
from rich.progress import track

from feature_extraction import EdgeFaceFeatureExtractor, FeatureExtractor

from .defaults import *

logger = logging.getLogger(__name__)


def parse_args():
    """Parses the arguments from the command line."""
    parser = argparse.ArgumentParser(description="Insert data into Milvus")
    parser.add_argument(
        "--data_path", type=str, help="Path to the root folder of the data", required=True
    )
    parser.add_argument("--uri", type=str, default=MILVUS_URI, help="Host of Milvus server")
    parser.add_argument(
        "--collection_name", type=str, default=MILVUS_DB_NAME, help="Collection name"
    )
    parser.add_argument("--db_name", type=str, default=MILVUS_COLLECTION_NAME, help="Database name")
    parser.add_argument(
        "--dimension", type=int, default=EDGE_FACE_DIM, help="Dimension of the vectors"
    )
    parser.add_argument("--metric_type", type=str, default=DEFAULT_METRIC, help="Metric type")
    parser.add_argument(
        "--delete_existing_collection", action="store_true", help="Delete existing collection"
    )

    return parser.parse_args()


def create_collection(
    client: MilvusClient,
    collection_name: str,
    vector_field_name: str,
    dimension: int,
    metric_type: str,
    delete_existing_collection: bool = False,
) -> None:
    """Creates a collection in Milvus.

    Args:
        client: MilvusClient object.
        collection_name (str): Name of the collection.
        vector_field_name (str): Name of the vector field.
        dimension (int): Dimension of the embeddings.
        metric_type (str): Metric type.
        delete_existing_collection (bool): Whether to delete the existing collection with the same name.
    """
    if collection_name in client.list_collections():
        if not delete_existing_collection:
            logger.warning(
                "Inserting data into an existing collection %s, run the script with the flag --delete_existing_collection for a fresh collection with the same name (destroying the existing one)."
                % collection_name
            )
        else:
            client.drop_collection(collection_name=collection_name)
            logger.info("Collection %s dropped." % collection_name)

    client.create_collection(
        collection_name=collection_name,
        vector_field_name=vector_field_name,
        dimension=dimension,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type=metric_type,
    )


def insert_lfw_data(
    data_path: Path,
    client: MilvusClient,
    collection_name: str,
    vector_field_name: str,
    feature_extractor: FeatureExtractor,
) -> None:
    """Inserts the data from a given path into the collection. The data is expected to be in the
    LFW format.

    The LFW dataset has a folder for each person, that contains all the images for that individual. This function
    only "inserts" the person_name/filename into the collection, which means that, for retrieval/searching, you will
    need to append the 'filename' in the collection to the root path of the LFW dataset.

    Args:
        data_path (Path): Path to the root folder of the data.
        client: MilvusClient object.
        collection_name (str): Name of the collection.
        vector_field_name (str): Name of the vector field.
    """
    images_paths = list(data_path.glob("**/*.jpg"))

    for image_path in track(
        images_paths, description=f"Inserting {len(images_paths)} images into Milvus"
    ):
        # read image
        image = cv2.imread(str(image_path))
        # extract embeddings
        embeddings = feature_extractor.predict(image)
        # insert into collection
        person_name = image_path.parent.name
        client.insert(
            collection_name=collection_name,
            data=[
                {
                    "filename": image_path.as_posix(),
                    "name": person_name,
                    vector_field_name: embeddings,
                }
            ],
        )


def main():
    """Main function to run the insert operation of all the data into Milvus."""
    args = parse_args()
    logger.info("Running insert operation with arguments: %s" % pprint.pp(args))

    # initialize the vector db client
    client = MilvusClient(uri=args.uri, db_name=args.db_name)

    # initialize the feature extractor
    extractor = EdgeFaceFeatureExtractor()

    # creating collection
    create_collection(
        client=client,
        collection_name=args.collection_name,
        vector_field_name=DEFAULT_VECTOR_FIELD_NAME,
        dimension=args.dimension,
        metric_type=args.metric_type,
        delete_existing_collection=args.delete_existing_collection,
    )

    # inserting data from the LFW dataset
    insert_lfw_data(
        data_path=Path(args.data_path),
        client=client,
        collection_name=args.collection_name,
        vector_field_name=DEFAULT_VECTOR_FIELD_NAME,
        feature_extractor=extractor,
    )


if __name__ == "__main__":
    main()
