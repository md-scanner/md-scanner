import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DeblurDataset  # custom dataset class
from model import DeblurCNN  # deblurring model class

# Define hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10
log_interval = 10

# Define data transformations
transform = transforms.Compose([
    # apply a random affine transformation to change the perspective of the images
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=255),
    transforms.ToTensor(),
])

# call dataset_generator.py to generate the dataset
os.system("python dataset_generator.py")

# Create instances of your custom dataset
train_dataset = DeblurDataset("dataset/", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define your neural network model (e.g., DeblurCNN)
model = DeblurCNN().cuda()

# Define loss function (e.g., Mean Squared Error)
criterion = nn.MSELoss()

# Define optimizer (e.g., Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (blurred_images, sharp_images) in enumerate(train_loader):
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(blurred_images.cuda())

        # Calculate the loss
        loss = criterion(outputs, sharp_images)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print training progress
        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save the model checkpoint after each epoch if needed
    checkpoint_path = f'model_checkpoint_epoch{epoch + 1}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved model checkpoint: {checkpoint_path}')

print('Training complete.')