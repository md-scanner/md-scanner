import random
import cv2 as cv
import pandas as pd
import torchvision.transforms.functional as F
import os
from os import path
from prepare_input import binarize_doc_image, adapt_char_image_size


script_dir = path.dirname(path.realpath(__file__))
test_data_dir = path.join(script_dir, "test_data")


class TestDocument:
    def __init__(self, img_file: str, box_file: str):
        self.img_file = img_file
        self.box_file = box_file

        self._load_image_file()
        self.bin_img = binarize_doc_image(self.img)

        self._load_box_file()


    def _load_image_file(self):
        img = cv.imread(self.img_file, cv.IMREAD_GRAYSCALE)
        img = F.to_tensor(img)
        self.img = img


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

            _, h, _ = self.img.shape

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

        prep_char_img = adapt_char_image_size(bin_char_img)

        return char_img, bin_char_img, prep_char_img, row


def sample_document():
    rand_file = path.join(test_data_dir, random.choice(os.listdir(test_data_dir)))
    rand_file = path.splitext(rand_file)[0]  # Trim extension

    img_file, box_file = rand_file + ".jpg", rand_file + ".box"
    return TestDocument(img_file, box_file)
