# Font Style Classifier (FSC)

## Pre-setup

The Font Style Classifier component strongly relies on [google/fonts](https://github.com/google/fonts) repository. 

Clone repository anywhere on your machine:
```
git clone https://github.com/google/fonts/ <google-fonts-dir>
```

## Scripts

Follows a small usage guide describing how to use our scripts. We assume your working directory is `font-style-classifier`.

First of all, you need to set the following environment variables:
```bash
export FSC_GOOGLE_FONTS_DIR=         # The path to the Google Fonts directory
export FSC_ENCODER_MODEL=            # One of: V1Net, SmallNet, TinyNet or VeryTinyNet
export FSC_ENCODER_CHECKPOINT_PATH=  # The training checkpoint to load
export FSC_DB_PATH=                  # The path to the qdrant database to generate 
export FSC_DB_COLLECTION_NAME=       # The name of the database collection to generate
export FSC_DATASET_CSV_PATH=         # The path to the dataset index .csv file
export FSC_DATASET_DIR=              # The path to the dataset folder
```

### Dataset generator: model/dataset-generator.py

This script is used to generate the dataset on which we train the FSC model and that we use to fill the database for retrieval.

**The dataset is required for most of the scripts.**

```
cd model
python3 dataset-generator.py
```

### Database generator: db_gen.py

This script is used to generate the database on which retrieval is performed.

**The database is required for most of the scripts.**

**NOTE: This script will fresh any existing DB.**

```
python3 db_gen.py [-f]
```

Options:
- `-f`: forcely delete the DB (without asking)


### Character classifier: classify.py

This script contains routines to classify a single character's font style (i.e. regular, italic or bold).

```
python3 classify.py
```

### Run inference on the encoder: model/inference.py

This script is used to run inference on the encoder and thus to roughly test its accuracy:

```
cd model
python3 inference.py
```

### Run inference on the encoder: eval.sh

This script is used to evaluate the 4 FSC models (V1Net, SmallNet, TinyNet, VeryTinyNet); writes the output result in `eval_results`.

```
bash eval.sh
```

## References:

Some links we found useful during the development:

- Lorem markdownum: https://github.com/jaspervdj/lorem-markdownum 
- Siamese network: https://builtin.com/machine-learning/siamese-network
