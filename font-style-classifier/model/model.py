import torch
from torch import nn
from torch.nn import functional as F


# References:
# - https://builtin.com/machine-learning/siamese-network
# - https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/



def count_learnable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class FSC_Encoder(nn.Module):
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
        """ Loads the given checkpoint onto the model. The checkpoint can be either a file or a loaded checkpoint. """

        if type(checkpoint) == str:
            checkpoint = torch.load(checkpoint)

        self.load_state_dict(checkpoint['model_state_dict'])


if __name__ == "__main__":
    model = FSC_Encoder()
    print(f"Learnable params: CNN: {count_learnable_params(model.cnn)}, FC: {count_learnable_params(model.fc)}, Total: {count_learnable_params(model)}")

    print("Inference test:")

    x = torch.randn(1, 1, 32, 32)
    print("x:", x.shape)
    
    y = model(x)
    print("y:", y.shape)



