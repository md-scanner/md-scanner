from math import floor, ceil
import random
from PIL import Image
import pandas as pd
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2 as cv


# OpenCV issue:
# https://stackoverflow.com/questions/52337870/python-opencv-error-current-thread-is-not-the-objects-thread/72090539#72090539


def prepare_input(char_img: Tensor):
    """
    Resizes the input character to fit the input of the FSC Encoder.
    The input is expected to be a 32x32 grayscale image.
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


TEST_DATA = [
    ("./test_data/img1.jpg", "./test_data/img1.box"),
    ("./test_data/img2.jpg", "./test_data/img2.box"),
]


def binarize_document_image(doc_img):
    _, bin_img = cv.threshold(doc_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return bin_img


if __name__ == "__main__":
    img_file, box_file = random.choice(TEST_DATA)

    print(f"Loading the image \"{img_file}\"...")
    img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)  # Load the whole document image

    # Binarize the document image
    bin_img = binarize_document_image(img)
    
    num_white_pixels = (bin_img == 255).sum()
    num_black_pixels = (bin_img.shape[0] * bin_img.shape[1]) - num_white_pixels

    if num_white_pixels < num_black_pixels:  # Invert, we want the background to be white!
        bin_img = 255 - bin_img

    # Show the result
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title("Original image")
    axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[0].axis('off')

    axs[1].set_title("Binarized image")
    axs[1].imshow(bin_img, cmap='gray', vmin=0, vmax=255)
    axs[1].axis('off')

    plt.show()

    h, w = bin_img.shape
    print(f"Image size: {bin_img.shape}")

    # Load the bounding box file and put it in a dataframe
    f = open(box_file, "r")
    lines = f.readlines()
    values = [line.split() for line in lines]
    df = pd.DataFrame(values, columns=['char', 'left', 'bottom', 'right', 'top', "_"])

    char = None
    while True:
        char = df.sample().iloc[0]  # Take a random character in [a-zA-Z0-9]
        if char['char'] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            break

    # Crop the character image out of the global one
    char_img = img[
        (h - int(char['top'])):(h - int(char['bottom'])),
        int(char['left']):int(char['right'])
        ]
    char_img = F.to_tensor(char_img)

    bin_char_img = bin_img[
        (h - int(char['top'])):(h - int(char['bottom'])),
        int(char['left']):int(char['right'])
        ]
    bin_char_img = F.to_tensor(bin_char_img)
    
    prepared_char_img = prepare_input(bin_char_img)

    # Show the result
    _, axs = plt.subplots(1, 3)

    axs[0].set_title("Cropped char")
    axs[0].imshow(char_img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    #axs[0].add_patch(plt.Rectangle((0, 0), char_img.shape[2], char_img.shape[1], fill=False, edgecolor='blue', linewidth=1))
    #axs[0].margins(0.001)
    axs[0].axis('off')

    axs[1].set_title("Binarized char")
    axs[1].imshow(bin_char_img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    #axs[1].add_patch(plt.Rectangle((0, 0), bin_char_img.shape[2], bin_char_img.shape[1], fill=False, edgecolor='blue', linewidth=1))
    #axs[1].margins(0.001)
    axs[1].axis('off')

    axs[2].set_title("Prepared char")
    axs[2].imshow(prepared_char_img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    #axs[2].add_patch(plt.Rectangle((0, 0), prepared_char_img.shape[2], prepared_char_img.shape[1], fill=False, edgecolor='blue', linewidth=1))
    #axs[2].margins(0.001)
    axs[2].axis('off')

    plt.show()

