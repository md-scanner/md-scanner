from common import *
import pandas as pd
from PIL import Image
from os import path
import torchvision.transforms.functional as F
from db_gen import get_complete_fonts
from db_retrieve import retrieve
import matplotlib.pyplot as plt
import itertools
from encoder.model import model
from math import *
from dataset import *

# Evaluate the font classification performance: given a character of the testing set, we evaluate whether the character
# is correctly classified as belonging to its font or not.

# Load the test set
dataset_df = get_dataset()
test_df = pd.read_csv(FSC_TEST_SET_CSV)
total_df_len = len(test_df)

test_df = test_df[test_df['font'].isin(get_complete_fonts())]
print(f"Extracted {len(test_df)}/{total_df_len} complete samples from the original test set")

N = 60


def _show_retrieved_samples(test_row, ret_samples):
    test_img_path = path.join(FSC_DATASET_DIR, test_row["filename"])
    print(f"Test image: {test_img_path}")

    test_img = plt.imread(test_img_path)

    num_grid_cols = 5
    num_grid_rows = ceil(N / num_grid_cols)

    fig, axs = plt.subplots(num_grid_rows, num_grid_cols + 1, figsize=(10, 8))

    for ax in itertools.chain(*axs):
        ax.axis('off')
        pass

    axs[0, 0].imshow(test_img, cmap='gray')
    axs[0, 0].text(0.5, -0.1,
                   test_row['font'] + "\n" +
                   ["Regular", "Bold", "Italic", "Bold/Italic"][test_row["is_italic"] * 2 + test_row["is_bold"]],
                   transform=axs[0, 0].transAxes,
                   ha='center', va='top',
                   backgroundcolor='white',
                   c='blue'
                   )

    for i, ret_sample in enumerate(ret_samples):
        payload = ret_sample['payload']
        img_path = path.join(FSC_DATASET_DIR, payload["filename"])
        img = plt.imread(img_path)

        img_x, img_y = i % num_grid_cols, i // num_grid_cols
        axs[img_y, img_x + 1].text(0.5, -0.1,
                                   f"{payload['char']} ({payload['font']})\n" +
                                   ["Regular", "Bold", "Italic", "Bold/Italic"][payload["is_italic"] * 2 + payload["is_bold"]] + "\n" +
                                   "{:.3f}".format(ret_sample['distance']),
                                   transform=axs[img_y, img_x + 1].transAxes,
                                   ha='center', va='top',
                                   backgroundcolor='white',
                                   c='blue'
                                   )
        axs[img_y, img_x + 1].imshow(img, cmap='gray')

    plt.tight_layout()
    plt.show()


def _test_sample_distance_to_font(test_sample, font=None):
    """ DEBUG function: tests the distance of the given test sample w.r.t. all of its font characters """

    test_img = load_dataset_image(test_sample['filename'])
    test_embedding = model(torch.unsqueeze(test_img, dim=0))

    if font is None:
        font = test_sample['font']

    def _font_slug(payload):
        style_str = ["", " b", " i", " bi"][payload["is_italic"] * 2 + payload["is_bold"]]
        return f"{payload['font']}{style_str}"

    for _, row in dataset_df[dataset_df['font'] == font].iterrows():
        x = load_dataset_image(row['filename'])
        x = torch.unsqueeze(x, dim=0)
        embedding = model(x)
        d = torch.norm(test_embedding - embedding)
        print(f"{test_sample['char']} ({_font_slug(test_sample)}) <-> {row['char']} ({_font_slug(row)}) => {d}")



def _main():
    while not test_df.empty:
        test_sample = test_df.sample()
        payload = test_sample.iloc[0]

        _test_sample_distance_to_font(payload, payload['font'])
        _test_sample_distance_to_font(payload, 'ibmplexsans')

        ret_samples = retrieve(load_dataset_image(payload['filename']), N)

        _show_retrieved_samples(payload, ret_samples)

        test_df.drop(test_sample.index)


if __name__ == "__main__":
    _main()

