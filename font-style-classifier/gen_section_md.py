from io import StringIO
import os
from subprocess import check_output
import uuid
from PIL import Image
import pandas as pd
from classify_word import ClassifyFontStyle
import sys
import time
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.utils import save_image


# ------------------------------------------------------------------------------------------------
# HocrToJsonConverter
# ------------------------------------------------------------------------------------------------


# TODO DELETE THIS SCRIPT!

# ------------------------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------------------------


class SectionMdGenerator:
    """
    Class responsible for generating the markdown of a section.
    A section is considered to be a rectangular area of the input document representing
    e.g. a paragraph, a title...
    """

    def __init__(self, section_img: Tensor, out_stream: StringIO):
        """
        :params:
            section_img:
                The Pytorch tensor representing the image of the section.
                Dimensions are expected to be (1, H, W).
            out_stream:
                The output stream where the markdown will be written.
                Usually, it's the output file.
        """

        self.section_img = section_img
        self.out_stream = out_stream

        self.bb_info_list = None
        self.num_words = None

        self.min_batch_size = 128
        self.next_word_id = 0

        self.cur_word_id = None  # The starting word ID of the current batch
        self.cur_batch = []


    def _write_md(self):
        words = []
        style_idx = -1


        def write_batch():
            nonlocal words, style_idx

            words_str = ' '.join(words)

            # Regular
            if style_idx == 0:
                self.out_stream.write(words_str)
            # Italic
            elif style_idx == 1:
                self.out_stream.write(f"*{words_str}*")
            # Bold
            elif style_idx == 2:
                self.out_stream.write(f"**{words_str}**")
            # The fuck (?)
            else:
                raise ValueError(f"Unknown style index for \"{word}\": {style_idx}")
            
            self.out_stream.write(" ")
            
            words = []
            style_idx = -1


        for word, word_style_idx in self.md_pre_output:
            if style_idx == -1:
                style_idx = word_style_idx
            elif style_idx != word_style_idx:
                write_batch()
                style_idx = word_style_idx

            words.append(word)


        write_batch()
        self.out_stream.write("\n")


    def generate(self):
        st = time.time()
        self._extract_bb_from_image()
        dt = time.time() - st
        print(f"[gen_section_md] Bounding boxes extracted")
        print(f"\tWords: {self.num_words}")
        print(f"\tTemp image: {self.tmp_img_path}")
        print(f"\tTemp .hocr: {self.tmp_hocr_path}")
        print(f"\tDT: {dt:.3f}")

        while True:
            st = time.time()
            if not self._load_next_char_batch():
                break
            dt = time.time() - st
            print(f"[gen_section_md] Batch prepared")
            print(f"\tWord range: {self.cur_word_id}/{self.num_words}-{self.next_word_id}/{self.num_words}")
            print(f"\tMin size: {self.min_batch_size}")
            print(f"\tBatch size: {len(self.cur_batch)}")
            print(f"\tDT: {dt:.3f}")

            st = time.time()
            self._classify_char_batch()
            dt = time.time() - st
            print(f"[gen_section_md] Batch classified")
            print(f"\tDT: {dt:.3f}")

            st = time.time()
            self._write_md()
            dt = time.time() - st
            print(f"[gen_section_md] Markdown written")
            print(f"\tDT: {dt:.3f}")

        print("[gen_section_md] DONE!")


# ------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input-image> <output-md|->")
        sys.exit(1)

    img_path = sys.argv[1]

    md_path = sys.argv[2]
    md_file = sys.stdout if md_path == '-' else open(md_path, "w")

    # Actually, it should be the image of a section
    doc_img = Image.open(img_path) \
          .convert('L')
    doc_img = F.to_tensor(doc_img)
    doc_img = binarize_doc_image(doc_img)

    md_gen = SectionMdGenerator(doc_img, md_file)
    bb_list = md_gen.generate()

