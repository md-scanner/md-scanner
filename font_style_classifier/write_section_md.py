from classify_section import classify_section
from torchvision.utils import save_image
import random


def write_section_md(img, out_stream):
    """ Given the image of a section, classifies every word's style and writes the Markdown.
    Args:
        img:
            A Pytorch tensor of shape (1, H, W) representing the grayscale image of the section.
        out_stream:
            The output stream onto which the Markdown is written.
    """

    words = []
    last_style = -1

    classified_words = classify_section(img)

    def write_batch():
        nonlocal words, last_style

        words_str = ' '.join(words)

        # Regular
        if last_style == 0:
            out_stream.write(words_str)
        # Italic
        elif last_style == 1:
            out_stream.write(f"*{words_str}*")
        # Bold
        elif last_style == 2:
            out_stream.write(f"**{words_str}**")
        # The fuck (?)
        else:
            raise ValueError(f"Unknown style index: {last_style}")

        out_stream.write(" ")

        words = []
        last_style = -1

    for word, font_style in classified_words:
        if last_style == -1:
            last_style = font_style
        elif last_style != font_style:
            write_batch()
            last_style = font_style

        words.append(word)

    if len(words) > 0:
        write_batch()
        out_stream.write("\n")

