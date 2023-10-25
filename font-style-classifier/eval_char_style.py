import random
import pandas as pd
from os import path
from classify import ClassifyFontStyle
from torchvision.io import read_image
import numpy as np
import torch
import sys
from os import environ as env


dataset = pd.read_csv(env['FSC_DATASET_CSV_PATH'])


def _filter_dataset(is_italic: bool, is_bold: bool):
    return dataset[(dataset['is_italic'] == is_italic) & (dataset['is_bold'] == is_bold)]


def _encode_style(italic: bool, bold: bool) -> int:
    if italic: # Italic
        return 1
    elif bold: # Bold
        return 2
    else: # Regular
        return 0


def _decode_style(style: int) -> bool:
    # TODO we need this function because in the dataset we have separate bools for italic/bold, we're
    # not using an index for the style
    return [
        (False, False), # Regular
        (True, False),  # Italic
        (False, True)   # Bold
    ][style]


def _calc_precision_recall_f1(tp, tn, fp, fn) -> float:
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * (p * r) / (p + r)
    return p, r, f1


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
            pick_same_style = round(random.random()) >= 0.5
            if pick_same_style:
                picked_style = _decode_style(style)  # Pick the same style
            else:
                picked_style = _decode_style(int(style + random.random() + 1) % 3)  # Pick another style
            
            # Query for a random character from the dataset
            sample_char = _filter_dataset(*picked_style).sample().iloc[0]
            
            sample_img_path = path.join(env['FSC_DATASET_DIR'], sample_char["filename"])
            sample_img = read_image(sample_img_path)  # Load a ByteTensor
            sample_img = sample_img.type(torch.FloatTensor) / 255.0  # Convert to FloatTensor

            # Fill up the batch
            batch += [(sample_img, sample_char["char"])]
            ground_truth += [_encode_style(sample_char["is_italic"], sample_char["is_bold"])]
        
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

        p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)

        print(f"[eval] Processed {i}/{NUM_SAMPLES} samples...")
        print(f"\tTP: {tp}")
        print(f"\tTN: {tn}")
        print(f"\tFP: {fp}")
        print(f"\tFN: {fn}")
        print(f"\tPrecision: {p:.3f}")
        print(f"\tRecall: {r:.3f}")
        print(f"\tF1-score: {f1:.3f}")

    return tp, tn, fp, fn


def main():
    if len(sys.argv) != 2:
        print(f"Invalid syntax: {sys.argv[0]} <out-csv>")
        sys.exit(1)

    with open(sys.argv[1], "w") as out_csv_file:
        out_csv_file.write(f"Style, TP, TN, FP, FN, Precision, Recall, F1-score\n")

        print(f"-" * 96)
        print(f"Regular char classification")
        print(f"-" * 96)

        tp, tn, fp, fn = eval_style_classification(0)
        p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)
        out_csv_file.write(f"Regular, {tp}, {tn}, {fp}, {fn}, {p:.3f}, {r:.3f}, {f1:.3f}\n")

        print(f"-" * 96)
        print(f"Italic char classification")
        print(f"-" * 96)

        tp, tn, fp, fn = eval_style_classification(1)
        p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)
        out_csv_file.write(f"Italic, {tp}, {tn}, {fp}, {fn}, {p:.3f}, {r:.3f}, {f1:.3f}\n")

        print(f"-" * 96)
        print(f"Bold char classification")
        print(f"-" * 96)

        tp, tn, fp, fn = eval_style_classification(2)
        p, r, f1 = _calc_precision_recall_f1(tp, tn, fp, fn)
        out_csv_file.write(f"Bold, {tp}, {tn}, {fp}, {fn}, {p:.3f}, {r:.3f}, {f1:.3f}\n")


if __name__ == "__main__":
    main()

