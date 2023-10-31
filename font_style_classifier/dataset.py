from os import path
from common import *
from PIL import Image
import torchvision.transforms.functional as F
import pandas as pd
import numpy as np


_dataset = None
_complete_fonts = None


def get_dataset_image_path(filename: str):
    return path.abspath(path.join(FSC_DATASET_DIR, filename))


def load_dataset_image(filename: str):
    img_path = path.join(FSC_DATASET_DIR, filename)
    img = Image.open(img_path)
    img = (F.pil_to_tensor(img) / 255.0).float()
    return img


def get_dataset():
    global _dataset

    if _dataset is not None:
        return _dataset

    print(f"[dataset] Loading dataset csv...")
    _dataset = pd.read_csv(FSC_DATASET_CSV)

    font_count = get_font_count()  # It's safe calling it here (no recursion)
    print(f"[dataset] Dataset loaded, contains {len(_dataset)} samples ({font_count} fonts)")

    return _dataset


def get_font_count():
    return len(get_dataset()['font'].unique())


def filter_dataset(**kwargs):
    df = get_dataset()

    _filter = np.ones(df.shape[0], dtype=bool)

    if 'font' in kwargs:
        _filter &= (df['font'] == kwargs['font'])

    if 'font_style' in kwargs:
        style_filters = [
            (~df['is_bold'] & ~df['is_italic']),  # Regular
            (df['is_bold'] & ~df['is_italic']),  # Bold
            (~df['is_bold'] & df['is_italic']),  # Italic
        ]
        _filter &= style_filters[kwargs['font_style']]

    if 'char' in kwargs:
        _filter &= (df['char'] == kwargs['char'])

    assert not df[_filter].empty

    return df[_filter]


def get_complete_fonts():
    """ Out of the dataset fonts, get those that have the three styles: Regular, Bold and Italic. """

    global _complete_fonts

    if _complete_fonts is not None:
        return _complete_fonts

    df = get_dataset()

    _complete_fonts = set()
    _complete_fonts.update(df[~df['is_bold'] & ~df['is_italic']]['font'].unique())  # Regular
    _complete_fonts = _complete_fonts.intersection(df[df['is_bold'] & ~df['is_italic']]['font'].unique())  # Bold
    _complete_fonts = _complete_fonts.intersection(df[df['is_italic'] & ~df['is_bold']]['font'].unique())  # Italic

    print(f"[dataset] Found {len(_complete_fonts)}/{get_font_count()} complete fonts (with Regular, Bold and Italic)")

    return _complete_fonts
