import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PIL import Image
import torchvision.transforms.functional as F
import pandas as pd
from model.model import FSC_Encoder

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


# For every dataset sample, run an inference and save the embedding to the DB
print("Reading dataset index...")
df = pd.read_csv(FSC_DATASET_CSV)

dataset_size = len(df)

for i, row in df.iterrows():
    print(f"{i + 1}/{dataset_size}\t{row['filename']}")

    img_path = os.path.join(FSC_DATASET_DIR, row["filename"])
    x = Image.open(img_path).convert('L')
    x = F.to_tensor(x)

    y = model(x.unsqueeze(0)).squeeze()

    result = db_client.upsert(
        collection_name=FSC_DB_COLLECTION_NAME,
        points=[
            PointStruct(
                id=i,
                vector=y.tolist(),
                payload=row.to_dict() 
            )
        ]
        )

