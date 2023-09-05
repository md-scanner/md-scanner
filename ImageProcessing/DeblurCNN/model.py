import torch
from torch import nn

class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        
        self.cnn = nn.Sequential(
            #1
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=19, padding="same"),
            nn.ReLU(),
            #2
            nn.Conv2d(in_channels=128, out_channels=320, kernel_size=1, padding="same"),
            nn.ReLU(),
            #3
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=1, padding="same"),
            nn.ReLU(),
            #4
            nn.Conv2d(in_channels=320, out_channels=320, kernel_size=1, padding="same"),
            nn.ReLU(),
            #5
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1, padding="same"),
            nn.ReLU(),
            #6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            #7
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1, padding="same"),
            nn.ReLU(),
            #8
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=5, padding="same"),
            nn.ReLU(),
            #9
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=5, padding="same"),
            nn.ReLU(),
            #10
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same"),
            nn.ReLU(),
            #11
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding="same"),
            nn.ReLU(),
            #12
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding="same"),
            nn.ReLU(),
            #13
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding="same"),
            nn.ReLU(),
            #14
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=7, padding="same"),
            nn.ReLU(),
            #15
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding="same"),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            # (2200, 1700, 3)
            nn.Linear(in_features=2200 * 1700 * 3, out_features=1024),
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

