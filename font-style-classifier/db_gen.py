import os
import time
import random
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PIL import Image
import torchvision.transforms.functional as F
import pandas as pd
from common import *
from encoder.model import model
import torch
import sys

print("[DbGen] Initializing the DB...")
db_client = QdrantClient(path=FSC_DB_PATH) 

# For every dataset sample, run an inference and save the embedding to the DB
print("[DbGen] Reading dataset index...")
dataset = pd.read_csv(FSC_DATASET_CSV)
dataset_length = len(dataset)

BATCH_SIZE = 100

batch = []

class DbGenerator:
    def __init__(self):
        self.cur_batch = []
        self.cur_batch_id = 0

        # Filter the original dataset to keep only fonts that have italic or bold styles
        total_fonts = dataset['font'].unique()
        complete_fonts = dataset[dataset['is_italic'] & dataset['is_bold']]['font'].unique()
        print(f"Found {len(complete_fonts)}/{len(total_fonts)} complete founds (with italic and bold)...")
        
        self.df = dataset[dataset['font'].isin(complete_fonts)]
        self.sample_count = len(self.df)
        print(f"Prepared {len(self.df)}/{len(dataset)} samples...")


    def _generate_embeddings(self):
        batched_x = torch.stack([img for img, _ in self.cur_batch])
        batched_x = batched_x.cuda()
        self.embeddings = model(batched_x)


    def _upsert_embeddings(self):
        db_client.upsert(
            collection_name=FSC_DB_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=random.getrandbits(64),
                    vector=embedding.tolist(),
                    payload=payload,
                )
                for (_, payload), embedding in zip(self.cur_batch, self.embeddings)
            ],
            wait=False
            )


    def _upsert_batch(self):
        sample_i = (self.cur_batch_id + 1) * BATCH_SIZE
        print(f"{sample_i}/{self.sample_count}, batch: {self.cur_batch_id} - ", end="")

        # Generate the embeddings for the current batch
        st = time.time()
        self._generate_embeddings()
        dt = time.time() - st

        print(f"Inference: {dt:.3f}, ", end="")

        # Insert the embeddings into the DB
        st = time.time()
        self._upsert_embeddings()
        el_count = db_client.count(collection_name=FSC_DB_COLLECTION_NAME).count
        dt = time.time() - st

        print(f"DB insertion: {dt:.3f} (count: {el_count}), ", end="")

        # Done!
        filling_dt = time.time() - self.filling_started_at
        print(f"Done! {filling_dt:.3f}")

        self.cur_batch = []
        self.cur_batch_id += 1


    def fill_db(self):
        self.cur_batch = []
        self.cur_batch_id = 0

        print("Filling DB...")

        self.filling_started_at = time.time()

        for _, row in self.df.iterrows():
            img_path = os.path.join(FSC_DATASET_DIR, row["filename"])
            img = Image.open(img_path)
            img = F.to_tensor(img)

            self.cur_batch.append((img, row.to_dict()))

            if len(self.cur_batch) >= BATCH_SIZE:
                self._upsert_batch()

        if len(self.cur_batch) > 0:
            self._upsert_batch()


def main():
    proceed = '-f' in sys.argv
    if not proceed:
        input("This action will fresh and regenerate the FSC_Database. Press any key to proceed...")

    # Recreate the collection (IMPORTANT: this will also delete old records!)
    db_client.recreate_collection(
        collection_name=FSC_DB_COLLECTION_NAME,
        vectors_config=VectorParams(size=128, distance=Distance.DOT),
        )
    
    db_gen = DbGenerator()
    db_gen.fill_db()


if __name__ == "__main__":
    main()
