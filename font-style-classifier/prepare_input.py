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


def binarize_doc_image(src_img: Tensor):
    assert src_img.shape[0] == 1
    assert src_img.dtype == torch.float32

    src_img_shape = src_img.shape

    # Convert the pytorch Tensor to a cv2 image (format: uint8), reference:
    # https://gist.github.com/gsoykan/369df298de35ecd9ec8253e28cd4ddbf
    src_img = src_img \
        .mul(255) \
        .to(dtype=torch.uint8) \
        .permute(1, 2, 0) \
        .numpy()

    _, out_img = cv.threshold(src_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    num_white_pixels = (out_img == 255).sum()
    num_black_pixels = (out_img.shape[0] * out_img.shape[1]) - num_white_pixels

    if num_white_pixels < num_black_pixels:  # Too much black: invert, we want the background to be white!
        out_img = 255 - out_img

    # Convert the cv2 image back to a pytorch Tensor
    out_img = torch.from_numpy(out_img) \
        .to(dtype=torch.float32) \
        .mul(1.0 / 255.0) \
        .unsqueeze(0) \
        .permute(0, 1, 2)

    assert out_img.shape == src_img_shape
    assert out_img.dtype == torch.float32

    return out_img


def adapt_char_image_size(char_img: Tensor):
    """
    Resizes the character to fit the input of the FSC Encoder.
    The input is expected to be a (1, 32, 32) tensor, representing a grayscale image.
    """

    assert char_img.shape[0] == 1

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

