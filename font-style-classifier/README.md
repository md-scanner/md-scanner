# FSC_Dataset

`FSC_Dataset` is the dataset we use to train `FSC_Encoder`. The dataset currently consists of >141000 32x32 images of black-on-white characters belonging to different fonts, and a `.csv` descriptor holding metadata.

### gen_db.py

The script used to generate `FSC_Dataset`.

Usage:

Clone [google/fonts](https://github.com/google/fonts) repository anywhere on your machine (`google-fonts-dir`):
```
git clone https://github.com/google/fonts/ <google-fonts-dir>
```

Set the following env variables:
```
FSC_DATASET_DIR=<dataset-dir>        # Where dataset images are saved
FSC_DATASET_CSV_PATH=<dataset-csv>   # Where .csv is written (default: <dataset-dir>/dataset.csv)
GOOGLE_FONTS_DIR=<google-fonts-dir>  # Path to google/fonts repository
```

Run the script:
```
python3 gen_db.py
```

# FSC_Model

# FSC_Database

`FSC_Database` is the database we use to store embeddings associated to characters rendered using different fonts and font styles (i.e. regular, bold and italic).
Embeddings are generated such that they're spatially near for characters belonging to the same font and far for characters of different fonts.

To store our embeddings we use [qdrant](https://github.com/qdrant/qdrant).

Install and run:
```
cd <font-style-classifier> 
docker pull qdrant/qdrant
docker run -p 6333:6333 -v <qdrant-db>:/qdrant/storage qdrant/qdrant
```

A database row consists of:
- `embedding`: a 1024-dimensional vector representing the character
- `payload`: metadata for the character, specifically:
    - `font`: the name of the font
    - `char`: a string representing the character
    - `is_italic`: a boolean telling whether the character is italic
    - `is_bold`: a boolean telling whether the character is bold

### gen_db.py

This script iterates over every sample of `FSC_Dataset`, generates the embedding for the character and inserts it into `FSC_Database` along with its metadata.

Usage:
```
python3 gen_db.py
```


### References:

- Random markdown generator: https://github.com/jaspervdj/lorem-markdownum 
- Siamese network: https://builtin.com/machine-learning/siamese-network

