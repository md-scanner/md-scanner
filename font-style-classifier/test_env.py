import random
import cv2 as cv
import pandas as pd
import torchvision.transforms.functional as F
from os import path

from prepare_input import prepare_input


script_dir = path.dirname(path.realpath(__file__))


TEST_DATA = [
    (path.join(script_dir, "./test_data/img1.jpg"), path.join(script_dir, "./test_data/img1.box")),
    (path.join(script_dir, "./test_data/img2.jpg"), path.join(script_dir, "./test_data/img2.box")),
]


class TestDocument:
    def __init__(self, img_file: str, box_file: str):
        self.img_file = img_file
        self.box_file = box_file

        self._load_image_file()
        self._binarize_image()

        self._load_box_file()


    def _load_image_file(self):
        self.img = cv.imread(self.img_file, cv.IMREAD_GRAYSCALE)


    def _binarize_image(self):
        _, bin_img = cv.threshold(self.img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
        num_white_pixels = (bin_img == 255).sum()
        num_black_pixels = (bin_img.shape[0] * bin_img.shape[1]) - num_white_pixels

        if num_white_pixels < num_black_pixels:  # Invert, we want the background to be white!
            bin_img = 255 - bin_img

        self.bin_img = bin_img


    def _load_box_file(self):
        f = open(self.box_file, "r")
        lines = f.readlines()
        values = [line.split() for line in lines]
        self.df = pd.DataFrame(values, columns=['char', 'left', 'bottom', 'right', 'top', "_"])


    def sample_char(self):
        # row
        # top, bottom, left, right

        while True:
            row = self.df.sample().iloc[0]  # Take a random character in [a-zA-Z0-9]

            if row['char'] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
                continue

            h, _ = self.img.shape

            top = h - int(row['top'])
            bottom = h - int(row['bottom'])
            left = int(row['left'])
            right = int(row['right'])

            if bottom <= top or right <= left:  # Validate crop coordinates
                continue

            break

        char_img = self.img[top:bottom, left:right]
        char_img = F.to_tensor(char_img)

        bin_char_img = self.bin_img[top:bottom, left:right]
        bin_char_img = F.to_tensor(bin_char_img)

        prep_char_img = prepare_input(bin_char_img)

        return char_img, bin_char_img, prep_char_img, row


def sample_document():
    img_file, box_file = random.choice(TEST_DATA)
    return TestDocument(img_file, box_file)

