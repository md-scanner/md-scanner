from io import StringIO
import os
from subprocess import check_output
import uuid
from PIL import Image
import pandas as pd
from html.parser import HTMLParser
from classify import ClassifyFontStyle
import sys
import time
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from prepare_input import binarize_doc_image, adapt_char_image_size


# ------------------------------------------------------------------------------------------------
# HocrToJsonConverter
# ------------------------------------------------------------------------------------------------

class HocrToJsonConverter(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []

        self.current_word = None
        self.current_char = None


    def _parse_bbox(self, bbox_str: str):
        return [int(x) for x in bbox_str.split(" ")[1:]]


    def _parse_conf(self, conf_str: str):
        return float(conf_str.split(" ")[1])


    def handle_starttag(self, tag, attrs):
        attrs = {key: value for key, value in attrs}
        if attrs is None:
            return
        
        if 'class' not in attrs:
            return

        if tag == "span" and attrs['class'] == 'ocrx_word':  # New word
            self.current_word = {
                "bbox": self._parse_bbox(attrs['title'].split("; ")[0]),
                "conf": self._parse_conf(attrs['title'].split("; ")[1]),
                "chars": []
            }
        elif tag == "span" and attrs['class'] == 'ocrx_cinfo':  # New character
            self.current_char = {
                "bbox": self._parse_bbox(attrs['title'].split("; ")[0]),
                "conf": self._parse_conf(attrs['title'].split("; ")[1]),
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


    def _extract_bb_from_image(self):
        tmp_img_name = str(uuid.uuid4())
        self.tmp_img_path = "/tmp/" + tmp_img_name + ".jpg"
        self.tmp_hocr_path = "/tmp/" + tmp_img_name + ".hocr"

        #print(f"Saving temporary image to: {tmp_img_path}")

        save_image(self.section_img, self.tmp_img_path)  # Save the image temporarily in /tmp/ so we can run tesseract

        #print(f"Running tesseract...")

        hocr_str = check_output([
            "tesseract",
            self.tmp_img_path,
            "-",
            "--psm", "6",
            "-c", "hocr_char_boxes=1",
            "-c", "tessedit_create_hocr=1"
            ]).decode('utf-8')

        with open(self.tmp_hocr_path, 'w') as f:
            f.write(hocr_str)

        #print(f"Parsing .hocr output...")

        json_converter = HocrToJsonConverter()
        json_converter.feed(hocr_str)

        # TODO Uncomment
        #os.remove(tmp_img_path)
        #os.remove(tmp_hocr_path)

        self.bb_info_list = json_converter.result
        self.num_words = len(self.bb_info_list)


    def _load_char_image(self, bbox: list):
        left, top, right, bottom = bbox

        # Crop the character out of the original document
        char_img = F.crop(self.section_img, top, left, bottom - top, right - left)
        char_img = adapt_char_image_size(char_img)  # Resize the image to be (1, 32, 32)

        assert char_img != None
        assert char_img.shape == (1, 32, 32)

        return char_img


    def _load_next_char_batch(self) -> bool:
        self.cur_word_id = self.next_word_id
        self.cur_batch = []

        while True:
            if self.next_word_id >= self.num_words or len(self.cur_batch) >= self.min_batch_size:
                break

            cur_word = self.bb_info_list[self.next_word_id]
            for char_bb_info in cur_word['chars']:
                self.cur_batch.append((
                    self._load_char_image(char_bb_info['bbox']),
                    char_bb_info['char'],
                    ))
            
            self.next_word_id += 1

        return len(self.cur_batch) > 0


    def _classify_char_batch(self):
        # Classify every character that was loaded into the batch
        classify = ClassifyFontStyle(self.cur_batch)
        classify()

        self.md_pre_output = []  # A list containing words and their styles, ready to be outputted

        # Now re-iterate the words, for each word we keep the style that is max voted
        batch_idx = 0
        for word in self.bb_info_list[self.cur_word_id:self.next_word_id]:
            word_style_indices = []
            for _ in word['chars']:
                style_idx = classify.result[batch_idx]
                word_style_indices.append(style_idx)
                batch_idx += 1

            word_txt = ''.join([x['char'] for x in word['chars']])

            print(f"Word: {word_txt}, Style: {word_style_indices}")

            # Keep the style that got the maximum vote among all the characters
            max_voted_style_idx = max(set(word_style_indices), key=word_style_indices.count)

            # TODO QUICK FIX
            max_voted_style_idx = max_voted_style_idx if max_voted_style_idx is None else max_voted_style_idx

            self.md_pre_output.append((word_txt, max_voted_style_idx,))


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

