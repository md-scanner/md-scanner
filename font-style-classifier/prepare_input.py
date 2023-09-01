from math import floor, ceil
import random
from PIL import Image
import pandas as pd
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2 as cv
import test_env


# OpenCV issue:
# https://stackoverflow.com/questions/52337870/python-opencv-error-current-thread-is-not-the-objects-thread/72090539#72090539


def prepare_input(char_img: Tensor):
    """
    Resizes the character to fit the input of the FSC Encoder.
    The input is expected to be a (1, 32, 32) tensor, representing a grayscale image.
    """

    _, h, w = char_img.shape

    # Resize such that the max side is 32
    if h > w:
        rh, rw = 32, int(32 * (w / h))
    else:
        rh, rw = int(32 * (h / w)), 32
    char_img = F.resize(char_img, size=(rh, rw), antialias=False)

    # Add padding to make it 32x32
    _, h, w = char_img.shape
    side = max(h, w)

    char_img = F.pad(
        char_img,
        padding=(
            floor((side - w) / 2.0),
            floor((side - h) / 2.0),
            ceil((side - w) / 2.0),
            ceil((side - h) / 2.0)
            ),
        fill=1
        )
    return char_img


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    test_doc = test_env.sample_document()

    # Show the result
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title("Image")
    axs[0].imshow(test_doc.img, cmap='gray', vmin=0, vmax=255)
    axs[0].axis('off')

    axs[1].set_title("Image (bin)")
    axs[1].imshow(test_doc.bin_img, cmap='gray', vmin=0, vmax=255)
    axs[1].axis('off')

    plt.show()

    char_img, bin_char_img, prep_char_img, _ = test_doc.sample_char()

    # Show the result
    _, axs = plt.subplots(1, 3)

    axs[0].set_title("Char")
    axs[0].imshow(char_img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    axs[0].axis('off')

    axs[1].set_title("Char (bin)")
    axs[1].imshow(bin_char_img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    axs[1].axis('off')

    axs[2].set_title("Char (bin+resize)")
    axs[2].imshow(prep_char_img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    axs[2].axis('off')

    plt.show()

