import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # To import common

import torch
from torch import nn
from torch.nn import functional as F
from common import *


# References:
# - https://builtin.com/machine-learning/siamese-network
# - https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/


def _count_learnable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# ------------------------------------------------------------------------------------------------
# V2 Network
# ------------------------------------------------------------------------------------------------

class V2Net(nn.Module):
    def __init__(self):
        super(FSC_Encoder, self).__init__()

        self.cnn = nn.Sequential(
            # 32x32:1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 16x16:128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 8x8:512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 4x4:512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 2x2:1024
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=8192),
            nn.ReLU(),
            nn.Linear(in_features=8192, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=128),
        )

    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

    def load_checkpoint(self, checkpoint):
        if type(checkpoint) == str:
            checkpoint = torch.load(checkpoint)

        self.load_state_dict(checkpoint['model_state_dict'])


# ------------------------------------------------------------------------------------------------
# V1 Network
# ------------------------------------------------------------------------------------------------


class V1Net(nn.Module):
    def __init__(self):
        super(V1Net, self).__init__()

        self.cnn = nn.Sequential(
            # 32x32:1
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 16x16:256
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 8x8:512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 4x4:512
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 512, out_features=4096),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=1024),
        )

    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

    def load_checkpoint(self, checkpoint):
        if type(checkpoint) == str:
            checkpoint = torch.load(checkpoint)

        self.load_state_dict(checkpoint['model_state_dict'])


# ------------------------------------------------------------------------------------------------
# Small Network
# ------------------------------------------------------------------------------------------------


class SmallNet(nn.Module):
    def __init__(self):
        super(FSC_Encoder, self).__init__()

        self.cnn = nn.Sequential(
            # 32x32:1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 16x16:128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 8x8:512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 4x4:512
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 2x2:1024
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
        )

    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

    def load_checkpoint(self, checkpoint):
        if type(checkpoint) == str:
            checkpoint = torch.load(checkpoint)

        self.load_state_dict(checkpoint['model_state_dict'])

# ------------------------------------------------------------------------------------------------
# Tiny Network
# ------------------------------------------------------------------------------------------------


class TinyNet(nn.Module):
    def __init__(self):
        super(FSC_Encoder, self).__init__()

        self.cnn = nn.Sequential(
            # 32x32:1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 16x16:64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 8x8:128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 4x4:256
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1024),
        )

    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

    def load_checkpoint(self, checkpoint):
        if type(checkpoint) == str:
            checkpoint = torch.load(checkpoint)

        self.load_state_dict(checkpoint['model_state_dict'])


# ------------------------------------------------------------------------------------------------
# Very Tiny Network
# ------------------------------------------------------------------------------------------------


class VeryTinyNet(nn.Module):
    def __init__(self):
        super(FSC_Encoder, self).__init__()

        self.cnn = nn.Sequential(
            # 32x32:1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 16x16:64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 8x8:128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 4x4:256
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=1024),
        )

    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    

    def load_checkpoint(self, checkpoint):
        if type(checkpoint) == str:
            checkpoint = torch.load(checkpoint)

        self.load_state_dict(checkpoint['model_state_dict'])


# ------------------------------------------------------------------------------------------------


#FSC_Encoder = V1Net
#FSC_Encoder = SmallNet
#FSC_Encoder = TinyNet
#FSC_Encoder = VeryTinyNet
FSC_Encoder = globals()[FSC_ENCODER_MODEL]

# Initialize a global model instance that can be used by other modules
print(f"[Model] Initializing the model \"{FSC_ENCODER_MODEL}\"...")
model = FSC_Encoder()

if os.path.exists(FSC_ENCODER_LATEST_CHECKPOINT):
    print(f"[Model] Loading latest checkpoint: \"{FSC_ENCODER_LATEST_CHECKPOINT}\"")
    model.load_checkpoint(FSC_ENCODER_LATEST_CHECKPOINT)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Learnable params: CNN: {_count_learnable_params(model.cnn)}, FC: {_count_learnable_params(model.fc)}, Total: {_count_learnable_params(model)}")

    print("Inference test:")

    x = torch.randn(1, 1, 32, 32)
    print("x:", x.shape)
    from time import time

    start = time()
    y = model(x)
    print(time()-start)
    print("y:", y.shape)

