import pandas as pd
from common import *


dataset = pd.read_csv(FSC_DATASET_CSV)


def split_dataset(dataset_df):
    fonts = dataset_df['font'].unique()

    # Test set
    print("Preparing test set...")
    test_df = pd.DataFrame()
    for font in fonts:  # Take 4 random samples per font
        test_df = pd.concat([test_df, dataset_df[dataset_df['font'] == font].sample(n=4)])
    dataset_df = dataset_df.drop(test_df.index)

    # Validation set
    print("Preparing validation set...")
    validation_df = pd.DataFrame()
    for font in fonts:  # Take other 4 random samples per font
        validation_df = pd.concat([validation_df, dataset_df[dataset_df['font'] == font].sample(n=4)])
    dataset_df = dataset_df.drop(validation_df.index)

    # Training set
    training_df = dataset_df

    return training_df, test_df, validation_df


print(f"Splitting dataset into training set, validation set, and test set...")
print(f"  Dataset: {len(dataset)} ({len(dataset['font'].unique())} fonts)")

training_set, test_set, validation_set = split_dataset(dataset)

print(f"Split dataset:")
print(f"  Training set: {len(training_set)} ({len(training_set['font'].unique())} fonts)")
print(f"  Test set: {len(test_set)} ({len(test_set['font'].unique())} fonts)")
print(f"  Validation set: {len(validation_set)} ({len(validation_set['font'].unique())} fonts)")

print(f"Merging tests (should be zero!):")
print(f"  Training set <-> Validation set: {len(pd.merge(training_set, validation_set))}")
print(f"  Validation set <-> Test set: {len(pd.merge(validation_set, test_set))}")
print(f"  Test set <-> Training set: {len(pd.merge(training_set, test_set))}")


training_set.to_csv(FSC_TRAINING_SET_CSV, index=False)
validation_set.to_csv(FSC_VALIDATION_SET_CSV, index=False)
test_set.to_csv(FSC_TEST_SET_CSV, index=False)
