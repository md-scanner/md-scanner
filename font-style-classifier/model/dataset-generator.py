from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
from os import path
import re
import csv


GOOGLE_FONTS_DIR="/work/cvcs_2023_group28/dataset-retriever/font-style-classifier/fonts"
FSC_DATASET_DIR="/work/cvcs_2023_group28/dataset-retriever/font-style-classifier/dataset"


FONT_BLACKLIST=[
    "zillaslabhighlight",
    "notocoloremojicompattest"
]


class DatasetGenerator:
    def __init__(self):
        self.num_generated_images = 0
        self.descriptor = {}


    def write_descriptor(self):
        with open(path.join(FSC_DATASET_DIR, "dataset.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["font", "char", "is_italic", "is_bold", 'filename'])

            for _, entry in self.descriptor.items():
                writer.writerow([
                    entry['font'],
                    entry['char'],
                    entry['is_italic'],
                    entry['is_bold'],
                    entry['filename']
                ])


    def _prepare_char_image(self, font, char):
        _, _, w, h = font.getbbox(char)

        char_img = Image.new('L', (w, h), color=0)

        draw = ImageDraw.Draw(char_img)
        draw.text(xy=(w / 2, h / 2), text=char, font=font, fill=255, anchor="mm")

        bbox = char_img.getbbox()
        char_img = char_img.crop(bbox)
        char_img = ImageOps.invert(char_img)

        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        side = max(w, h)
        if h > w:
            offset = (side - w) // 2, 0
        else:
            offset = 0, (side - h) // 2

        img = Image.new('L', (max(w, h), max(w, h)), color=255)
        img.paste(char_img, offset)

        img = img.resize((32, 32))

        return img


    def generate_characters_for_font_style(self, ttf_file, font_id, is_italic, is_bold):
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        if not path.exists(FSC_DATASET_DIR):
            os.mkdir(FSC_DATASET_DIR)

        print(f"{font_id}: ", end="")

        font = ImageFont.truetype(ttf_file, size=32)
        
        for char in chars:
            try:
                char_img_filename = f"{font_id}-{char}-{'i' if is_italic else ''}{'b' if is_bold else ''}.png"
                char_img_path = path.join(FSC_DATASET_DIR, char_img_filename)

                if path.exists(char_img_path):
                    print("_", end="")
                else:
                    img = self._prepare_char_image(font, char)
                    img.save(char_img_path)

                    self.num_generated_images += 1

                    print(char, end="")

            except Exception as e:
                # To ignore:
                # - OSError: execution context too long

                if isinstance(e, KeyboardInterrupt):
                    raise e

                print("!", end="")
                continue

            # The char was generated
            self.descriptor[char_img_filename] = {
                'font': font_id,
                'char': char,
                'is_italic': is_italic,
                'is_bold': is_bold,
                'filename': char_img_filename,
            }
        
        print(f" ({self.num_generated_images} images)")


    def generate(self):
        self.num_generated_images = 0
        self.descriptor = {}

        filename_rgx = re.compile(r'(?:^|\s+)filename: \"([^\"]*)\"', re.IGNORECASE)
        style_rgx = re.compile(r'(?:^|\s+)style: \"([^\"]*)\"', re.IGNORECASE)
        weight_rgx = re.compile(r'(?:^|\s+)weight: ([^\s]*)', re.IGNORECASE)

        ofl_fonts_dir = path.join(GOOGLE_FONTS_DIR, "ofl")

        for font_dirname in os.listdir(ofl_fonts_dir):
            if not path.exists(path.join(ofl_fonts_dir, font_dirname, "METADATA.pb")):
                continue

            if font_dirname in FONT_BLACKLIST:
                continue

            with open(path.join(ofl_fonts_dir, font_dirname, "METADATA.pb")) as f:
                metadata = f.read()

                filename_matches = filename_rgx.findall(metadata)
                style_matches = style_rgx.findall(metadata)
                weight_matches = weight_rgx.findall(metadata)

                num_styles = len(filename_matches)

                if num_styles != len(style_matches) and num_styles != len(weight_matches):
                    continue

                for i in range(num_styles):
                    filename = filename_matches[i]
                    style = style_matches[i]
                    weight = int(weight_matches[i])

                    is_italic = style == "italic"
                    is_bold = weight > 500

                    ttf_file = path.join(ofl_fonts_dir, font_dirname, filename)

                    font_id = font_dirname.replace("-", "_")
                    self.generate_characters_for_font_style(ttf_file, font_id, is_italic, is_bold)


        print(f"Generated {self.num_generated_images} images, writing .csv descriptor (entries: {len(self.descriptor)})...")
        self.write_descriptor()


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate()

