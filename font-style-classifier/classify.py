import torch
from torch import Tensor
from model.model import FSC_Encoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


FSC_DB_PATH="./.fsc-db"
FSC_DB_COLLECTION_NAME="embeddings"

print("Loading model...")
model = FSC_Encoder()
model.load_checkpoint("./model/latest-checkpoint.pt")

print("Initializing the DB...")
db_client = QdrantClient(path=FSC_DB_PATH)


def classify(img: Tensor):
    x = torch.unsqueeze(img, 0)
    y = model(x)
    y = torch.squeeze(y)
   
    hits = db_client.search(
        collection_name=FSC_DB_COLLECTION_NAME,
        query_vector=y.tolist(),
        with_vectors=True,
        limit=100
        )


if __name__ == "__main__":
    img = torch.randn((1,32,32))
    classify(img)

    db_client.close()
    pass

