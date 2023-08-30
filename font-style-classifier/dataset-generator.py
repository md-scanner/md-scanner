from PIL import Image, ImageDraw, ImageFont
import os
from os import path
import re


GOOGLE_FONTS_DIR="/tmp/fonts/"
FSC_DATASET_DIR="/tmp/fsc-dataset"


FONT_BLACKLIST=[
    "zillaslabhighlight",
]


num_generated_images = 0


def generate_characters_for_font_style(ttf_file, font_id, is_italic, is_bold):
    global num_generated_images

    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    if not path.exists(FSC_DATASET_DIR):
        os.mkdir(FSC_DATASET_DIR)

    print(f"{font_id}: ", end="")

    for char in chars:
        char_filename = path.join(FSC_DATASET_DIR, f"{font_id}-{char}-{'i' if is_italic else ''}{'b' if is_bold else ''}.png")
        char_filename_exists = path.exists(char_filename)

        print('_' if char_filename_exists else char, end="")

        if char_filename_exists:
            continue

        img = Image.new('RGB', (32, 32), color=(255, 255, 255))
        font = ImageFont.truetype(ttf_file, size=24)

        char_size = font.getsize(char)
        char_pos = ((32 - char_size[0]) / 2, (32 - char_size[1]) / 2)

        d = ImageDraw.Draw(img)
        d.text(xy=char_pos, text=char, font=font, fill=(0, 0, 0))
        img.save()

        num_generated_images += 1

    print(f" ({num_generated_images} images)")


def generate():
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
                generate_characters_for_font_style(ttf_file, font_id, is_italic, is_bold)


if __name__ == "__main__":
    generate()

