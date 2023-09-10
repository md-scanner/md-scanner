#!/usr/bin/env bash

mkdir -p eval_results

SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

echo "------------------------------------------------------------------------------------------------"
echo "V1Net"
echo "------------------------------------------------------------------------------------------------"

export FSC_ENCODER_MODEL="V1Net"
export FSC_ENCODER_CHECKPOINT_PATH="./model/latest-checkpoint-v1.pt"
export FSC_DB_PATH="./.v1-db"
export FSC_DB_COLLECTION_NAME="embeddings"
export FSC_DATASET_CSV_PATH="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
export FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset"

if [ ! -d $FSC_DB_PATH ]; then
    python3 $SCRIPT_DIR/gen_db.py -f
fi
#python3 $SCRIPT_DIR/eval_char.py eval_results/${FSC_ENCODER_MODEL}_char_eval.csv
#python3 $SCRIPT_DIR/eval_word.py eval_results/${FSC_ENCODER_MODEL}_word_eval.csv

echo "------------------------------------------------------------------------------------------------"
echo "SmallNet"
echo "------------------------------------------------------------------------------------------------"

export FSC_ENCODER_MODEL="SmallNet"
export FSC_ENCODER_CHECKPOINT_PATH="./model/latest-checkpoint-small.pt"
export FSC_DB_PATH="./.small-db"
export FSC_DB_COLLECTION_NAME="embeddings"
export FSC_DATASET_CSV_PATH="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
export FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset"

if [ ! -d $FSC_DB_PATH ]; then
    python3 $SCRIPT_DIR/gen_db.py -f
fi
python3 $SCRIPT_DIR/eval_char.py eval_results/${FSC_ENCODER_MODEL}_char_eval.csv
#python3 $SCRIPT_DIR/eval_word.py eval_results/${FSC_ENCODER_MODEL}_word_eval.csv

echo "------------------------------------------------------------------------------------------------"
echo "TinyNet"
echo "------------------------------------------------------------------------------------------------"

export FSC_ENCODER_MODEL="TinyNet"
export FSC_ENCODER_CHECKPOINT_PATH="./model/latest-checkpoint-tiny.pt"
export FSC_DB_PATH="./.tiny-db"
export FSC_DB_COLLECTION_NAME="embeddings"
export FSC_DATASET_CSV_PATH="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
export FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset"

if [ ! -d $FSC_DB_PATH ]; then
    python3 $SCRIPT_DIR/gen_db.py -f
fi
python3 $SCRIPT_DIR/eval_char.py eval_results/${FSC_ENCODER_MODEL}_char_eval.csv
#python3 $SCRIPT_DIR/eval_word.py eval_results/${FSC_ENCODER_MODEL}_word_eval.csv

echo "------------------------------------------------------------------------------------------------"
echo "VeryTinyNet"
echo "------------------------------------------------------------------------------------------------"

export FSC_ENCODER_MODEL="VeryTinyNet"
export FSC_ENCODER_CHECKPOINT_PATH="./model/latest-checkpoint-verytiny.pt"
export FSC_DB_PATH="./.verytiny-db"
export FSC_DB_COLLECTION_NAME="embeddings"
export FSC_DATASET_CSV_PATH="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv"
export FSC_DATASET_DIR="/home/rutayisire/unimore/cv/md-scanner/fsc-dataset"

if [ ! -d $FSC_DB_PATH ]; then
    python3 $SCRIPT_DIR/gen_db.py -f
fi
python3 $SCRIPT_DIR/eval_char.py eval_results/${FSC_ENCODER_MODEL}_char_eval.csv
#python3 $SCRIPT_DIR/eval_word.py eval_results/${FSC_ENCODER_MODEL}_word_eval.csv
