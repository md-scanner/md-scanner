# compare markdown files in input directories using Levenshtein distance, BLUE score, and METEOR

import sys
import os
import re
import numpy as np
from nltk import edit_distance
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python markdown-comparer.py <input_dir1> <reference_dir>")
    sys.exit(1)

input_dir1 = sys.argv[1]
reference_dir = sys.argv[2]


def get_files(input_dir):
    files = []
    for file in os.listdir(input_dir):
        if file.endswith("md"):
            files.append(file)
    return files

def compare_files(file1, file2):
    with open(file1, "r") as f1:
        text1 = f1.read()
    with open(file2, "r") as f2:
        text2 = f2.read()
    return compare_text(text1, text2)

df = pd.DataFrame(columns=["Levenshtein distance", "BLEU score", "METEOR score"])

def compare_text(text1, text2):
    # calculate and output Levenshtein distance
    dist = edit_distance(text1, text2)
    

    split1 = word_tokenize(text1)
    split2 = word_tokenize(text2)

    # calculate and output BLEU score
    bleu = sentence_bleu([split1], split2)
    print("BLEU score: {}".format(bleu), flush=True)
    

    # calculate and output METEOR score
    meteor = single_meteor_score(split1, split2)
    print("METEOR score: {}".format(meteor), flush=True)
    df.loc[len(df)] = [dist, bleu, meteor]
    return dist, bleu, meteor


files = get_files(input_dir1)

avg_dist, avg_bleu, avg_meteor = (0, 0, 0)

for i in range(len(files)):
    hypothesis = os.path.join(input_dir1, files[i])
    to_compare = os.path.join(reference_dir, files[i].split("-")[0])+".md"
    print("Comparing {} and {}".format(hypothesis, to_compare), flush=True)
    dist, bleu, meteor = compare_files(hypothesis, to_compare)
    avg_dist += dist
    avg_bleu += bleu
    avg_meteor += meteor


df.to_csv("all-scores-ours.csv")
print("Average Levenshtein distance: {}".format(avg_dist/len(files)), flush=True)
print("Average BLEU score: {}".format(avg_bleu/len(files)), flush=True)
print("Average METEOR score: {}".format(avg_meteor/len(files)), flush=True)

    