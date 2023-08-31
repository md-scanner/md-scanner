import torch
from torch import nn
from torch.nn import functional as F


# References:
# - https://builtin.com/machine-learning/siamese-network
# - https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/


class FSC_Encoder(nn.Module):
    def __init__(self):
        super(FSC_Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool4 = nn.MaxPool2d((2, 2))


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = x.flatten(1)

        return x


    def load_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint['model_state_dict'])

