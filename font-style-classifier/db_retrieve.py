from common import *
from qdrant_client import QdrantClient
from encoder.model import model
from qdrant_client import models
import itertools

print(f"[DbRet] Initializing the DB...")
db_client = QdrantClient(path=FSC_DB_PATH)

el_count = db_client.count(collection_name=FSC_DB_COLLECTION_NAME).count
print(f"[DbRet] DB initialized, {el_count} elements")


def retrieve_with_embedding(embedding, count: int, skip_first=False):
    query_results = db_client.search(
        collection_name=FSC_DB_COLLECTION_NAME,
        query_vector=embedding.tolist(),
        with_vectors=True,
        limit=count
    )
    query_results = [  # Mask out qdrant data structures
        {'distance': result.score, 'embedding': result.vector, 'payload': result.payload}
        for result in query_results
    ]
    if skip_first:
        query_results = query_results[1:]
    return query_results


def retrieve(char_image, num_retrieved_samples: int, skip_first=False):
    """ Retrieves the nearest samples from DB to the given query sample.
    The results are returned in descent order.
    """

    embedding = model(torch.unsqueeze(char_image, dim=0))
    embedding = torch.squeeze(embedding)
    return retrieve_with_embedding(embedding, num_retrieved_samples, skip_first=skip_first)


def aggregate_retrieved_results(word_retrieved_samples) -> str:
    """ Given a list of lists of retrieval result (one retrieval result per word's character); aggregates them in order
    to get one response: the final font classification. """

    # Flatten out the retrieved samples for every word's character
    _list = itertools.chain(*word_retrieved_samples)
    nearest_fonts = [sample['payload']['font'] for sample in _list]
    return max(set(nearest_fonts), key=nearest_fonts.count)

