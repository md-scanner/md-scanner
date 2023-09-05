import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class DeblurDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform

        # List all image files in the blurred subfolder
        self.image_files = [f for f in os.listdir(os.path.join(dataset_folder, 'sharp')) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load blurred and sharp images
        blurred_image_path = os.path.join(self.dataset_folder, 'blurred', 'blurred_'+self.image_files[idx])
        sharp_image_path = os.path.join(self.dataset_folder, 'sharp', self.image_files[idx])

        blurred_image = Image.open(blurred_image_path)
        sharp_image = Image.open(sharp_image_path)

        # Apply transformations if specified
        if self.transform:
            blurred_image = self.transform(blurred_image)
            sharp_image = self.transform(sharp_image)

        # Convert images to tensors
        blurred_image = torch.tensor(blurred_image).float()
        sharp_image = torch.tensor(sharp_image).float()

        return blurred_image, sharp_image