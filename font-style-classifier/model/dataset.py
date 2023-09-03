import random
from torch.utils.data import Dataset
from torchvision.transforms import v2
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Reference:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# ------------------------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------------------------

class FSC_Dataset:
    def __init__(self, dataset_csv: str, root_dir: str, epoch_dim=1024):
        self.df = pd.read_csv(dataset_csv)
        self.root_dir = root_dir
        self.epoch_dim = epoch_dim

        self.fonts = self.df['font'].unique()

        self.augment = v2.Compose([
            v2.RandomAffine(
                degrees=0,
                scale=(0.7, 1.0),
                fill=255
            ),
            v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()])
            #v2.GaussianBlur(3, (0.01, 1.2)),
        ])


    def _load_sample(self, df_row):
        img = Image.open(path.join(self.root_dir, df_row['filename']))
        img = self.augment(img)
        return img


    def pick_random_font(self):
        return random.choice(self.fonts)


    def pick_same_font_input(self, font=None):
        if not font:
            font = self.pick_random_font()
         
        x1 = self.df[self.df['font'] == font].sample().iloc[0]
        x2 = self.df[self.df['font'] == font].sample().iloc[0]

        return self._load_sample(x1), self._load_sample(x2), 1.0


    def pick_diff_font_input(self, font=None):
        if not font:
            font = self.pick_random_font()

        x1 = self.df[self.df['font'] == font].sample().iloc[0]
        x2 = self.df[self.df['font'] != font].sample().iloc[0]

        return self._load_sample(x1), self._load_sample(x2), 0.0


    def __getitem__(self, i):  # The index is not used, we pick the sample randomly!
        sf = round(random.random())
        if sf >= 0.5:
            return self.pick_same_font_input()
        else:
            return self.pick_diff_font_input()


    def __len__(self):
        return self.epoch_dim


def show_dataset_sampling(dataset):
    while True:
        fig, axs = plt.subplots(1, 2)
        
        x1, x2, sf = dataset[0]  # The index isn't relevant

        plt.title("sf: " + str(sf))

        axs[0].imshow(x1.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1.0)
        axs[1].imshow(x2.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1.0)

        plt.show()


# ------------------------------------------------------------------------------------------------


FSC_DATASET_DIR = "/work/cvcs_2023_group28/dataset-retriever/font-style-classifier/dataset"
FSC_DATASET_CSV = path.join(FSC_DATASET_DIR, "dataset.csv")


if __name__ == "__main__":
    dataset = FSC_Dataset(FSC_DATASET_CSV, root_dir=FSC_DATASET_DIR, epoch_dim=0)

    show_dataset_sampling(dataset)

    #show_data_augmentations(dataset)
