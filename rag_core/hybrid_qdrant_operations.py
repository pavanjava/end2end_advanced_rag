import json
import os

from qdrant_client import QdrantClient, models
from utils.decorators import compute_execution_time
from dotenv import load_dotenv, find_dotenv
from typing import List, Any


class HybridQdrantOperations:
    _ = load_dotenv(find_dotenv())

    def __init__(self):
        self.payload_path = "../data.json"
        self.collection_name = "hybrid-multi-stage-queries-collection"
        self.DENSE_MODEL_NAME = "snowflake/snowflake-arctic-embed-s"
        self.SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
        # collect to our Qdrant Server
        self.client = QdrantClient(url=os.environ['QDRANT_API_BASE'], api_key=os.environ['QDRANT_API_KEY'])
        self.client.set_model(self.DENSE_MODEL_NAME)
        # comment this line to use dense vectors only
        self.client.set_sparse_model(self.SPARSE_MODEL_NAME)
        self.metadata = []
        self.documents = []

    @compute_execution_time
    def load_data(self):
        with open(self.payload_path) as fd:
            for line in fd:
                obj = json.loads(line)
                self.documents.append(obj.pop("description"))
                self.metadata.append(obj)

    @compute_execution_time
    def create_collection(self):

        if not self.client.collection_exists(collection_name=f"{self.collection_name}"):
            self.client.create_collection(
                collection_name=f"{self.collection_name}",
                vectors_config=self.client.get_fastembed_vector_params(),
                # comment this line to use dense vectors only
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(on_disk=True),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0,
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                ),
                shard_number=4
            )

    @compute_execution_time
    def insert_documents(self):
        self.client.add(
            collection_name=self.collection_name,
            documents=self.documents,
            metadata=self.metadata,
            parallel=5,  # Use all available CPU cores to encode data if the value is 0
        )
        self._optimize_collection_after_insert()

    @compute_execution_time
    def hybrid_search(self, text: str, top_k: int = 5) -> List[dict[str, Any]]:
        # self.client.query will have filters also if you want to do query on filter data.
        search_result = self.client.query(
            collection_name=self.collection_name,
            query_text=text,
            limit=top_k,  # 5 the closest results
        )
        # `search_result` contains found vector ids with similarity scores
        # along with the stored payload

        # Select and return metadata
        metadata = [hit.metadata for hit in search_result]
        return metadata

    @compute_execution_time
    def _optimize_collection_after_insert(self):
        self.client.update_collection(
            collection_name=f'{self.collection_name}',
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=30000)
        )


# only to be used as a driver code else just use the class from else where.
# if __name__ == '__main__':
    # ops = HybridQdrantOperations()
    # only run the below for the first time when you newly create collection and want to ingest the data.
    # ops.load_data()
    # ops.create_collection()
    # ops.insert_documents()

    # results = ops.hybrid_search(text="What are the gaming companies in bangalore?", top_k=10000)
    # print(results)
