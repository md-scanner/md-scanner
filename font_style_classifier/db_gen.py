import time
import random
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus
from PIL import Image
import torchvision.transforms.functional as F
from common import *
from encoder.model import model
import torch
import sys
from dataset import get_dataset, get_complete_fonts, load_dataset_image

# For every dataset sample, run an inference and save the embedding to the DB

BATCH_SIZE = 100

batch = []

dataset = get_dataset()
complete_fonts = get_complete_fonts()


class DbGenerator:
    def __init__(self):
        self.db_client = QdrantClient(path=FSC_DB_PATH)

        self.cur_batch = []
        self.cur_batch_id = 0

        self.df = dataset[dataset['font'].isin(complete_fonts)]
        self.sample_count = len(self.df)

        self.point_id = 0

        print(f"Prepared {len(self.df)}/{len(dataset)} samples...")

    def _recreate_collection(self):
        self.db_client.recreate_collection(
            collection_name=FSC_DB_COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=Distance.EUCLID),
        )

    def _generate_embeddings(self):
        x = torch.stack([img for img, _ in self.cur_batch])
        self.embeddings = model(x)

    def _upsert_embeddings(self):
        points = []
        for (_, payload), embedding in zip(self.cur_batch, self.embeddings):
            points.append(PointStruct(
                id=random.getrandbits(64),
                vector=embedding.tolist(),
                payload=payload,
            ))

        op_result = self.db_client.upsert(
            collection_name=FSC_DB_COLLECTION_NAME,
            points=points,
            wait=True
            )
        assert op_result.status == UpdateStatus.COMPLETED

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
        el_count = self.db_client.count(collection_name=FSC_DB_COLLECTION_NAME).count
        dt = time.time() - st

        print(f"DB insertion: {dt:.3f} (count: {el_count}), ", end="")

        # Done!
        filling_dt = time.time() - self.filling_started_at
        print(f"Done! {filling_dt:.3f}")

        self.cur_batch = []
        self.cur_batch_id += 1

    def regenerate(self):
        self._recreate_collection()

        self.cur_batch = []
        self.cur_batch_id = 0

        print("Filling DB...")

        self.filling_started_at = time.time()

        for _, row in self.df.iterrows():
            img = load_dataset_image(row['filename'])
            self.cur_batch.append((img, row.to_dict()))

            if len(self.cur_batch) >= BATCH_SIZE:
                self._upsert_batch()

        if len(self.cur_batch) > 0:
            self._upsert_batch()


def main():
    proceed = '-f' in sys.argv
    if not proceed:
        input("This action will fresh and regenerate the FSC_Database. Press any key to proceed...")

    db_gen = DbGenerator()
    db_gen.regenerate()


if __name__ == "__main__":
    main()
