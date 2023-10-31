from classify_section import classify_section
from PIL import Image
import torchvision.transforms.functional as F
from common import *
import numpy as np
import random


def _main():
    files = list(os.listdir(FSC_CLASSIFY_DATASET_DIR))
    random.shuffle(files)

    for f in files:
        if not f.endswith(".jpg"):
            continue

        gt_str = f.split("-")[0]
        gt = ["regular", "bold", "italic"].index(gt_str)

        doc_img_path = path.join(FSC_CLASSIFY_DATASET_DIR, f)
        doc_img = Image.open(doc_img_path).convert("L")
        doc_img = (F.pil_to_tensor(doc_img) / 255.0).float()

        result = classify_section(doc_img)

        matched = np.sum(np.array([x for _, x in result]) == gt)
        total = len(result)

        print(f"Document \"{f}\", GT: {gt_str}, Matched: {matched}/{total}")


if __name__ == "__main__":
    _main()
