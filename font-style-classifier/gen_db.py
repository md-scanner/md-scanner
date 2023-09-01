import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PIL import Image
import torchvision.transforms.functional as F
import pandas as pd
from model.model import FSC_Encoder
import torch

FSC_DB_PATH="./.fsc-db"
FSC_DB_COLLECTION_NAME="embeddings"

FSC_DATASET_CSV="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset"

print("Initializing the DB...")
db_client = QdrantClient(path=FSC_DB_PATH) 

db_client.recreate_collection(
    collection_name=FSC_DB_COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )


# Load the model and initialize it to the latest training checkpoint
print("Loading the model...")
model = FSC_Encoder()
model.load_checkpoint("./model/latest-checkpoint.pt")
model = model.cuda()

# For every dataset sample, run an inference and save the embedding to the DB
print("Reading dataset index...")
df = pd.read_csv(FSC_DATASET_CSV)

dataset_size = len(df)

BATCH_SIZE = 128

batch = []

def upsert_batch(batch_id: int):
    print(f"Processing batch #{batch_id}...")

    batched_x = torch.stack(batch)
    batched_x = batched_x.cuda()
    batched_y = model(batched_x)

    print(f"Upserting point: ", end="")
    for j, y in enumerate(batched_y):
        row_id = batch_id * 128 + j
        print(f"{row_id}, ", end="")

        db_client.upsert(
            collection_name=FSC_DB_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=row_id,
                    vector=y.tolist(),
                    payload=df.iloc[row_id].to_dict() 
                )
            ]
            )
    print()


for i, row in df.iterrows():
    img_path = os.path.join(FSC_DATASET_DIR, row["filename"])
    img = Image.open(img_path)
    x = F.to_tensor(img)
    batch.append(x)

    if len(batch) >= BATCH_SIZE:
        upsert_batch()
        batch = []

if len(batch) > 0:
    upsert_batch()
