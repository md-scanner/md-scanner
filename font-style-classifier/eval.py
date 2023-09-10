import os
import time
import random
import pandas as pd
from os import path
from classify import ClassifyFontStyle
from torchvision.io import read_image
from torchvision.utils import save_image
import numpy as np
import torch


FSC_DB_PATH="./.fsc-db"
FSC_DB_COLLECTION_NAME="embeddings"

FSC_DATASET_CSV="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset"

dataset = pd.read_csv(FSC_DATASET_CSV)


def _get_style_idx(italic: bool, bold: bool) -> int:
    if italic: # Italic
        return 1
    elif bold: # Bold
        return 2
    else: # Regular
        return 0
    

def _filter_dataset(is_italic: bool, is_bold: bool):
    return dataset[(dataset['is_italic'] == is_italic) & (dataset['is_bold'] == is_bold)]


def _decode_style(style: int) -> bool:
    # TODO we need this function because in the dataset we have separate bools for italic/bold, we're
    # not using an index for the style
    return [
        (False, False), # Regular
        (True, False),  # Italic
        (False, True)   # Bold
    ][style]


def eval_style_classification(style: int):
    BATCH_SIZE = 128
    NUM_SAMPLES = 10000

    tp, tn, fp, fn = 0, 0, 0, 0

    i = 0
    while True:
        batch = []
        ground_truth = []

        for _ in range(0, min(BATCH_SIZE, NUM_SAMPLES - i)):
            # Pick a character belonging to the given `style` with a 50% probability
            if round(random.random()) >= 0.5:
                picked_style = _decode_style(style)  # Pick the same style
            else:
                picked_style = _decode_style(int(style + random.random() + 1) % 3)  # Pick another style
            
            # Query for a random character from the dataset
            sample_char = _filter_dataset(*picked_style).sample().iloc[0]
            
            sample_img_path = path.join(FSC_DATASET_DIR, sample_char["filename"])
            sample_img = read_image(sample_img_path)
            sample_img = sample_img.type(torch.FloatTensor)    # convert to FloatTensor

            save_image(sample_img, "/tmp/sample.png")

            # Fill up the batch
            batch += [(sample_img, sample_char["char"])]
            ground_truth += [_get_style_idx(sample_char["is_italic"], sample_char["is_bold"])]
        
        i += len(batch)

        if len(batch) > 0:
            # Perform the classification
            classify = ClassifyFontStyle(batch)
            result = classify()

            result = np.array(result)
            ground_truth = np.array(ground_truth)

            # Evaluate the results
            tp += (result[ground_truth == style] == style).sum()
            tn += (result[ground_truth != style] != style).sum()
            fp += (result[ground_truth != style] == style).sum()
            fn += (result[ground_truth == style] != style).sum()
        else:
            break

        p = tp / (tp + fp)
        r = tp / (tp + fn)

        print(f"[eval] Processed {i}/{NUM_SAMPLES} samples...")
        print(f"\tTP: {tp}")
        print(f"\tTN: {tn}")
        print(f"\tFP: {fp}")
        print(f"\tFN: {fn}")
        print(f"\tPrecision: {p:.3f}")
        print(f"\tRecall: {r:.3f}")
        print(f"\tF1-score: {2 * (p * r) / (p + r):.3f}")


if __name__ == "__main__":
    print(f"-" * 96)
    print(f"Regular classification evaluation")
    print(f"-" * 96)

    eval_style_classification(0)

    print(f"-" * 96)
    print(f"Italic classification evaluation")
    print(f"-" * 96)

    eval_style_classification(1)

    print(f"-" * 96)
    print(f"Bold classification evaluation")
    print(f"-" * 96)

    eval_style_classification(2)

