import pandas as pd
from common import *


dataset = pd.read_csv(FSC_DATASET_CSV)


def split_dataset(dataset, training_frac, test_frac):
    assert training_frac + test_frac < 1.0

    training_set = dataset.sample(frac=training_frac)
    dataset = dataset.drop(training_set.index)

    test_set = dataset.sample(frac=test_frac)
    validation_set = dataset.drop(test_set.index)

    return training_set, test_set, validation_set


print(f"Splitting dataset into training set, validation set, and test set...")
print(f"  Dataset: {len(dataset)} ({len(dataset['font'].unique())} fonts)")

training_set, test_set, validation_set = \
    split_dataset(dataset, training_frac=0.6, test_frac=0.25)  # validation_frac=

print(f"Splitted:")
print(f"  Training set: {len(training_set)} ({len(training_set['font'].unique())} fonts)")
print(f"  Test set: {len(test_set)} ({len(test_set['font'].unique())} fonts)")
print(f"  Validation set: {len(validation_set)} ({len(validation_set['font'].unique())} fonts)")

training_set.to_csv(FSC_TRAINING_SET_CSV, index=False)
validation_set.to_csv(FSC_VALIDATION_SET_CSV, index=False)
test_set.to_csv(FSC_TEST_SET_CSV, index=False)
