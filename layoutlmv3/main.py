import argparse
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
import pytesseract

def main():
    image_path = "test.jpg"
    model_path = "microsoft/layoutlmv3-base"

    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    model.eval()

    tokenizer = LayoutLMv3Tokenizer.from_pretrained(model_path)

    image = Image.open(image_path)
    tesseract_output = pytesseract.image_to_boxes(image)

    print(tesseract_output)

    with torch.no_grad():
        output = model(image)

    print(output.shape)

main()
