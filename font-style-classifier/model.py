import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import FSC_Dataset
import time
from datetime import datetime
import os
from os import path


script_dir = path.dirname(path.realpath(__file__))


if not torch.cuda.is_available():
    raise Exception("CUDA is not available!")


# References:
# - https://builtin.com/machine-learning/siamese-network
# - https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, x1, x2, y):
        dw = torch.norm(x1 - x2, dim=1)
        loss = (y * 0.5 * (dw ** 2)) + ((1 - y) * 0.5 * torch.clamp(self.margin - dw, min=0.0) ** 2)
        return torch.mean(loss)


# How many items are uploaded to the GPU in parallel
BATCH_SIZE = 64

# The dimension of an epoch in terms of iterations (i.e. the number of batch to draw from the dataset)
EPOCH_DIM = 128

# After how much time log the training progress
TRAINING_LOG_DELAY = 5.0


class Trainer:
    def __init__(self, model):
        self.model = model
        self.model.cuda()

        self.loss_fn = ContrastiveLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)

        self.dataset = FSC_Dataset(
            "/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv",
            "/home/rutayisire/unimore/cv/md-scanner/fsc-dataset",
            epoch_dim=EPOCH_DIM * BATCH_SIZE
            )
        self.dataset_loader = DataLoader(self.dataset, batch_size=BATCH_SIZE)

        self.iter = 0
        self.epoch = 0

        self.running_loss_iter = 0  # How many iterations have been done since the last log
        self.running_loss = 0.0  # The sum of the losses since the last log
        self.last_loss = 0.0  # The last loss logged
        
        self.started_at = time.time()
        self.last_log_time = 0


    def _elapsed_time(self):
        return time.time() - self.started_at


    def run_train_one_epoch(self):
        self.model.train(True)

        for data in self.dataset_loader:
            x1, x2, sf = data
            x1, x2, sf = x1.cuda(), x2.cuda(), sf.cuda()

            self.optimizer.zero_grad()

            y1 = self.model(x1)
            y2 = self.model(x2)

            loss = self.loss_fn(y1, y2, sf)
            loss.backward()

            self.optimizer.step()

            self.running_loss += loss.item()

            if (time.time() - self.last_log_time) > TRAINING_LOG_DELAY and self.running_loss_iter > 0:
                self.last_loss = self.running_loss / self.running_loss_iter
                self.running_loss = 0.0
                self.running_loss_iter = 0

                elapsed_time = time.time() - self.started_at
                print(f'TRA - Epoch: {self.epoch}, Iter: {self.iter}, Average loss: {self.last_loss}, Elapsed time: {elapsed_time:.2f}')

                self.last_log_time = time.time()


            self.running_loss_iter += 1
            self.iter += 1

        return self.last_loss


    def run_validation(self):
        self.model.eval()  # self.model.train(False)

        with torch.no_grad():
            # Pick a random font
            font = self.dataset.pick_random_font()

            # The sum of the distances of inputs of the same font.
            # We expect this to decrease over training
            sf_sum = 0.0

            # The sum of the distances of inputs of diffeerent fonts.
            # We expect this to increase over training
            df_sum = 0.0

            num_validation_iters = 128

            for _ in range(num_validation_iters):
                x1, x2, _ = self.dataset.pick_same_font_input(font)
                x1, x2 = x1.cuda(), x2.cuda()

                y1 = self.model(x1)
                y2 = self.model(x2)

                d = torch.norm(y1 - y2)
                sf_sum += d

            for _ in range(num_validation_iters):
                x1, x2, _ = self.dataset.pick_diff_font_input(font)
                x1, x2 = x1.cuda(), x2.cuda()

                y1 = self.model(x1)
                y2 = self.model(x2)

                d = torch.norm(y1 - y2)
                df_sum += d
            
        print(f"VAL - Num iters: {num_validation_iters}, SF sum: {sf_sum:.3f}, DF sum: {df_sum:.3f}")


    def save_checkpoint(self):
        checkpoint_filename = f"checkpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
        checkpoint_dir = path.join(script_dir, ".checkpoints")

        # Create .checkpoints directory if not exists
        if not path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        torch.save({
            'epoch': self.epoch,
            'iter': self.iter,
            'elapsed_time': self._elapsed_time(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.last_loss
        }, path.join(checkpoint_dir, checkpoint_filename))

        print(f"CHK - Saved checkpoint: {checkpoint_filename}")


    def train(self):
        print(f"Running training, batch size: {BATCH_SIZE}...")

        self.epoch = 0
        while True:
            print(f"-" * 96)
            print(f"EPOCH {self.epoch} (dim: {EPOCH_DIM})")
            print(f"-" * 96)
            
            self.run_train_one_epoch()
            self.run_validation()
            self.save_checkpoint()

            self.epoch += 1


if __name__ == "__main__":
    print(f"Pytorch CUDA Version is: {torch.version.cuda}")

    model = Encoder()

    trainer = Trainer(model)
    trainer.train()
