from common import *
from qdrant_client import QdrantClient
from encoder.model import model
from qdrant_client import models

print(f"[DbRet] Initializing the DB...")
db_client = QdrantClient(path=FSC_DB_PATH)

el_count = db_client.count(collection_name=FSC_DB_COLLECTION_NAME).count
print(f"[DbRet] DB initialized, {el_count} elements")


def _retrieve_with_embedding(embedding, count: int):
    print(f"[retrieve] Retrieving {count} elements...")
    query_results = db_client.search(
        collection_name=FSC_DB_COLLECTION_NAME,
        query_vector=embedding.tolist(),
        with_vectors=True,
        limit=count,
        search_params=models.SearchParams(exact=True)
    )
    query_results = [  # Mask out qdrant data structures
        {'distance': result.score, 'embedding': result.vector, 'payload': result.payload}
        for result in query_results
    ]
    return query_results


def retrieve(char_image, num_retrieved_samples: int):
    """ Retrieves the nearest samples from DB to the given query sample.
    The results are returned in descent order.
    """

    embedding = model(torch.unsqueeze(char_image, dim=0))
    embedding = torch.squeeze(embedding)
    return _retrieve_with_embedding(embedding, num_retrieved_samples)


def aggregate_retrieved_results(retrieved_samples) -> str:
    """ Given a list of retrieval results ordered in descent order (nearest to farthest), aggregates them to get a
    final font classification."""

    nearest_fonts = [sample['payload']['font'] for sample in retrieved_samples]
    return max(set(nearest_fonts), key=nearest_fonts.count)


def retrieve_one(char_image, num_retrieved_samples=10):
    return aggregate_retrieved_results(retrieve(char_image, num_retrieved_samples))
