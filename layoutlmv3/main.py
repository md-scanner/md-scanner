from transformers import AutoProcessor, AutoModelForTokenClassification
from datasets import load_dataset

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")

example = dataset[0]
image = example["image"]
words = example["tokens"]
boxes = example["bboxes"]
word_labels = example["ner_tags"]

#image.save("example0.png")

# image 762x1000
# words 145
# boxes 145
# word_labels 145

encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

# encoding['input_ids'].shape (1, 208)
# encoding['attention_mask'].shape (1, 208)
# encoding['bbox'].shape (1, 208, 4)

outputs = model(**encoding)
loss = outputs.loss
logits = outputs.logits

# logits.shape (1, 208, 7)

res = processor.tokenizer.decode([0])
