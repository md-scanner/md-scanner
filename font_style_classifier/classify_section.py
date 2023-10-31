import subprocess
from html.parser import HTMLParser
import cv2 as cv
import uuid
from time import time
from PIL import Image
import matplotlib.pyplot as plt
import random
from common import *
from math import *
from dataset import load_dataset_image, filter_dataset
from classify_word import classify_word_font, classify_word_style
import torchvision.transforms.functional as F
from torchvision.utils import save_image


class HocrToJsonConverter(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []

        self.current_word = None
        self.current_char = None

    def handle_starttag(self, tag, attrs):
        attrs = {key: value for key, value in attrs}
        if attrs is None:
            return

        if 'class' not in attrs:
            return

        def parse_bbox(bbox_str: str):
            return [int(x) for x in bbox_str.split(" ")[1:]]

        def parse_conf(conf_str: str):
            return float(conf_str.split(" ")[1])

        if tag == "span" and attrs['class'] == 'ocrx_word':  # New word
            self.current_word = {
                "bbox": parse_bbox(attrs['title'].split("; ")[0]),
                "conf": parse_conf(attrs['title'].split("; ")[1]),
                "chars": []
            }
        elif tag == "span" and attrs['class'] == 'ocrx_cinfo':  # New character
            self.current_char = {
                "bbox": parse_bbox(attrs['title'].split("; ")[0]),
                "conf": parse_conf(attrs['title'].split("; ")[1]),
            }

    def handle_endtag(self, tag):
        if self.current_char:
            self.current_word['chars'].append(self.current_char)
            self.current_char = None
        elif self.current_word:
            self.result.append(self.current_word)
            self.current_word = None

    def handle_data(self, data):
        if self.current_char:
            self.current_char['char'] = data


def _binarize_doc_image(img):
    """ Binarizes the given document image in a way such that the background is white and the text is black. """

    assert img.shape[0] == 1, f"Unexpected image shape: {img.shape}"
    assert img.dtype == torch.float32, f"Unexpected image dtype: {img.dtype}"

    src_img_shape = img.shape

    # Convert the pytorch Tensor to a cv2 image (format: uint8), reference:
    # https://gist.github.com/gsoykan/369df298de35ecd9ec8253e28cd4ddbf
    src_img = img \
        .mul(255) \
        .to(dtype=torch.uint8) \
        .permute(1, 2, 0) \
        .cpu() \
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


def _extract_bb_list(img):
    """ Given an image, extracts the bounding boxes for characters, split in words.
    Returns:
        A list with one entry corresponding to one word of lists: one entry per character.
    """

    tmp_img_name = str(uuid.uuid4())
    tmp_img_path = "/tmp/" + tmp_img_name + ".jpg"
    tmp_hocr_path = "/tmp/" + tmp_img_name + ".hocr"

    save_image(img, tmp_img_path)  # Save the image temporarily in /tmp/ so we can run tesseract

    hocr_str = subprocess.check_output([
        "tesseract",
        tmp_img_path,
        "-",
        "--psm", "6",
        "-c", "hocr_char_boxes=1",
        "-c", "tessedit_create_hocr=1"
    ]).decode('utf-8')

    with open(tmp_hocr_path, 'w') as f:
        f.write(hocr_str)

    json_converter = HocrToJsonConverter()
    json_converter.feed(hocr_str)

    os.remove(tmp_img_path)
    os.remove(tmp_hocr_path)

    bounding_boxes = json_converter.result
    return bounding_boxes


def _resize_image_to_encoder_input(char_img):
    """ Resizes the character to fit the input of the FSC Encoder.
    The input is expected to be a (1, 32, 32) tensor, representing a grayscale image.
    """

    assert char_img.shape[0] == 1

    _, h, w = char_img.shape

    # Resize such that the max side is 32
    if h > w:
        rh, rw = 32, int(32 * (w / h))
    else:
        rh, rw = int(32 * (h / w)), 32
    char_img = F.resize(char_img, size=[rh, rw], antialias=False)

    # Add padding to make it 32x32
    _, h, w = char_img.shape
    side = max(h, w)

    char_img = F.pad(
        char_img,
        padding=[
            floor((side - w) / 2.0),
            floor((side - h) / 2.0),
            ceil((side - w) / 2.0),
            ceil((side - h) / 2.0)
        ],
        fill=1
    )

    return char_img


def _load_and_resize_char_image(img, bbox: list):
    left, top, right, bottom = bbox

    # Crop the character out of the original document
    char_img = F.crop(img, top, left, bottom - top, right - left)
    char_img = _resize_image_to_encoder_input(char_img)  # Resize the image to be (1, 32, 32)

    assert char_img is not None
    assert char_img.shape == (1, 32, 32)

    return char_img


def classify_section(img):
    """ Extracts words out of the given image and classifies every word's style.
    Returns:
        A list of (word_string, font_style)
    """

    bin_img = _binarize_doc_image(img)

    result = []

    print("[classify_section] Extracting BBs...")
    bb_list = _extract_bb_list(bin_img)

    for i, word_info in enumerate(bb_list):
        word_txt = "".join(x['char'] for x in word_info['chars'])
        print(f"[classify_section] Word \"{word_txt}\" {i}/{len(bb_list)}", end="")

        word = []

        # Load word's characters from dataset
        start_at = time()
        for char_info in word_info['chars']:
            if char_info['char'].isalnum():
                char_img = _load_and_resize_char_image(bin_img, char_info['bbox'])
                word.append((char_img, char_info['char']))
        print(f", Chars loaded in {time() - start_at:.3f}", end="")

        if len(word) > 0:
            # Classify word style
            start_at = time()
            pred_font_style = classify_word_style(word)
            print(f", Classified as \"{font_style_str(pred_font_style)}\" in {time() - start_at:.3f}", end="")

            word_str = "".join(x for _, x in word)
            result.append((word_str, pred_font_style))

        print()

    return result


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


def _display_doc_vs_bin_doc(doc_img, bin_img):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.axis("off")
    ax1.imshow(torch.squeeze(doc_img).cpu(), cmap="gray")

    ax2.axis("off")
    ax2.imshow(torch.squeeze(bin_img).cpu(), cmap="gray")

    plt.tight_layout()
    plt.show()


def _display_word_classification(word, style_gt: str, font: str, pred_font_style=None):
    fig, axs = plt.subplots(4, len(word), squeeze=False)

    word_txt = "".join([x for _, x in word])

    fig.suptitle(f"Word: \"{word_txt}\" ({style_gt}), Font: \"{font}\"")

    for i, (char_img, char) in enumerate(word):
        axs[0, i].axis('off')
        axs[0, i].imshow(torch.squeeze(char_img).cpu(), cmap="gray")
        axs[0, i].text(0.5, -0.1, char,
                       transform=axs[0, i].transAxes,
                       ha='center', va='top'
                       )

        for j in range(0, 3):
            if i == 0:
                txt_color = 'green' if j == pred_font_style else "black"
                bg_color = 'yellow' if j == pred_font_style else "white"

                style_txt = ["Regular", "Bold", "Italic"][j]
                axs[j + 1, i].text(0.5, -0.1, style_txt, backgroundcolor=bg_color, c=txt_color, ha="center",
                                   transform=axs[j + 1, i].transAxes)

            char_img_path = filter_dataset(font=font, font_style=j, char=char).iloc[0]['filename']
            # print(f"Loading character image at: \"{char_img_path}\"")

            char_img = load_dataset_image(char_img_path)
            axs[j + 1, i].axis('off')
            axs[j + 1, i].imshow(torch.squeeze(char_img).cpu(), cmap="gray")

    plt.tight_layout()
    plt.show()


def _main():
    font_style = random.choice(['regular', 'bold', 'italic'])
    files = [
        path.join(FSC_CLASSIFY_DATASET_DIR, f)
        for f in os.listdir(FSC_CLASSIFY_DATASET_DIR)
        if f.startswith(font_style) and f.endswith(".jpg")
    ]

    doc_img_path = random.choice(files)
    doc_img = Image.open(doc_img_path).convert("L")
    doc_img = (F.pil_to_tensor(doc_img) / 255.0).float()

    bin_img = _binarize_doc_image(doc_img)
    _display_doc_vs_bin_doc(doc_img, bin_img)

    bb_list = _extract_bb_list(bin_img)
    for word_info in bb_list:
        word = []

        for char_info in word_info['chars']:
            if char_info['char'].isalnum():
                char_img = _load_and_resize_char_image(bin_img, char_info['bbox'])
                word.append((char_img, char_info['char']))

        if len(word) > 0:
            best_font = classify_word_font(word)
            _display_word_classification(word, font_style, best_font)

            pred_font_style = classify_word_style(word)
            _display_word_classification(word, font_style, best_font, pred_font_style)


if __name__ == "__main__":
    _main()
