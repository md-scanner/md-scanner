import random
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
from os import path
from PIL import Image
import matplotlib.pyplot as plt

# Reference:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


FSC_DATASET_DIR="/tmp/fsc-dataset"

# ------------------------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------------------------

class FontStyleClassifierDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filenames = sorted(os.listdir(self.root_dir))
        self.transform = None

    def get_similar_sample(self, font):
        pass

    def get_dissimilar_sample(self, font):
        pass

    def __getitem__(self, i):
        img = Image.open(path.join(self.root_dir, self.filenames[i]))

        # TODO TRANSFORM
        if not self.transform:
            pass

        filename_no_ext = self.filenames[i].split('.')[0]
        (font_id, char, font_style) = filename_no_ext.split('-')

        annotated_sample = {
            'image': img,
            "filename": self.filenames[i],
            'font': font_id,
            'char': char,
            'italic': 'i' in font_style,
            'bold': 'b' in font_style,
        }
        return annotated_sample

# ------------------------------------------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------------------------------------------

augmentations = v2.Compose([
    v2.RandomAffine(
        degrees=20,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        fill=(255, 255, 255)
    ),
    v2.GaussianBlur(3, (0.01, 1.2)),
])

transforms = v2.RandomApply([augmentations], 0.3)

def show_data_augmentations(dataset):
    while True:
        fig, axs = plt.subplots(1, 2)

        i = random.randint(0, len(dataset))
        
        sample = dataset[i]
        
        img = sample['image']
        axs[0].set_title(sample['filename'])
        axs[0].imshow(img)

        transformed_img = augmentations(img)
        axs[1].set_title(sample['filename'] + " (transformed)")
        axs[1].imshow(transformed_img)

        plt.show()

# ------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    dataset = FontStyleClassifierDataset(root_dir=FSC_DATASET_DIR)
    show_data_augmentations(dataset)

