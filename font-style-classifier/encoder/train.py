import os
from os import path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # To import common

import time
from datetime import datetime
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model import FSC_Encoder
from common import *

import torchvision
torchvision.disable_beta_transforms_warning()

from dataset import FSC_Dataset


script_dir = path.dirname(path.realpath(__file__))


if not torch.cuda.is_available():
    raise Exception("CUDA is not available!")


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, x1, x2, y):
        dw = torch.norm(x1 - x2, dim=1)

        loss1 = dw ** 2
        loss2 = torch.clamp(self.margin - dw, min=0.0) ** 2

        glob_loss = (y * 0.5 * loss1) + ((1 - y) * 0.5 * loss2)
        return torch.mean(glob_loss)


# How many items are uploaded to the GPU in parallel
BATCH_SIZE = 400

# The dimension of an epoch in terms of iterations (i.e. the number of batches to draw from the dataset)
EPOCH_DIM = 100

# After how much time log the training progress
TRAINING_LOG_DELAY = 5.0

# checkpoint-20230831005021.pt

class Trainer:
    def __init__(self, model: FSC_Encoder, load_latest_checkpoint=False):
        self.model = model
        self.model.cuda()

        self.loss_fn = ContrastiveLoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            50,
        )

        self.training_set = FSC_Dataset(FSC_TRAINING_SET_CSV, epoch_dim=EPOCH_DIM * BATCH_SIZE)
        self.training_set_loader = DataLoader(self.training_set, batch_size=BATCH_SIZE)

        self.validation_set = FSC_Dataset(FSC_VALIDATION_SET_CSV, epoch_dim=EPOCH_DIM * BATCH_SIZE)
        self.validation_set_loader = DataLoader(self.validation_set, batch_size=BATCH_SIZE)

        self.iter = 0
        self.epoch = 0

        self.running_loss_iter = 0  # How many iterations have been done since the last log
        self.running_loss = 0.0  # The sum of the losses since the last log
        self.last_loss = 0.0  # The last loss logged
        
        self.started_at = time.time()
        self.last_log_time = 0

        if load_latest_checkpoint:
            self._load_checkpoint()


    def _load_checkpoint(self) -> bool:
        checkpoint_dir = path.join(script_dir, ".checkpoints")
        latest_filename = path.join(checkpoint_dir, "latest.pt")
        
        if not path.exists(latest_filename):
            print("Latest checkpoint not found:", latest_filename, flush=True)
            return False

        print("Loading latest checkpoint:", latest_filename, flush=True)

        checkpoint = torch.load(latest_filename)

        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']
        self.model.load_checkpoint(checkpoint)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return True


    def _elapsed_time(self):
        return time.time() - self.started_at


    def _run_train_one_epoch(self):
        self.model.train(True)

        for data in self.training_set_loader:
            x1, x2, sf = data
            x1, x2, sf = x1.cuda(), x2.cuda(), sf.cuda()

            self.optimizer.zero_grad()

            y1 = self.model(x1)
            y2 = self.model(x2)

            loss = self.loss_fn(y1, y2, sf)
            loss.backward()

            self.optimizer.step()

            self.running_loss += loss.item()
            self.running_loss_iter += 1

            if (time.time() - self.last_log_time) > TRAINING_LOG_DELAY:
                self.loss = self.running_loss / self.running_loss_iter
                self.running_loss = 0.0
                self.running_loss_iter = 0

                lr = self.scheduler.get_last_lr()

                elapsed_time = time.time() - self.started_at
                print(f'TRA - Epoch: {self.epoch}, Iter: {self.iter}, Loss: {self.loss}, LR: {lr}, Elapsed time: {elapsed_time:.2f}', flush=True)

            self.iter += 1
        self.scheduler.step()

        return self.loss


    def _run_validation(self):
        self.model.eval()  # self.model.train(False)

        with torch.no_grad():
            # Pick a random font
            font = self.validation_set.pick_random_font()

            # The sum of the distances of inputs of the same font.
            # We expect this to decrease over training
            sf_mean_dist = 0.0

            # The sum of the distances of inputs of different fonts.
            # We expect this to increase over training
            df_mean_dist = 0.0

            num_validation_iters = 128

            for _ in range(num_validation_iters):
                x1, x2, _ = self.validation_set.pick_same_font_input(font)
                x1, x2 = x1.cuda(), x2.cuda()  # Move to GPU
 
                # Insert one dimension for the batch
                x1, x2 = torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)

                y1 = self.model(x1)
                y2 = self.model(x2)

                d = torch.norm(y1 - y2)
                sf_mean_dist += d

            for _ in range(num_validation_iters):
                x1, x2, _ = self.validation_set.pick_diff_font_input(font)
                x1, x2 = x1.cuda(), x2.cuda()  # Move to GPU
 
                # Insert one dimension for the batch
                x1, x2 = torch.unsqueeze(x1, 0), torch.unsqueeze(x2, 0)

                y1 = self.model(x1)
                y2 = self.model(x2)

                d = torch.norm(y1 - y2)
                df_mean_dist += d
        
        sf_mean_dist /= num_validation_iters
        df_mean_dist /= num_validation_iters

        print(f"VAL - Num iters: {num_validation_iters}, SF sum: {sf_mean_dist:.3f}, DF sum: {df_mean_dist:.3f}", flush=True)


    def _save_checkpoint(self):
        checkpoint_filename = f"checkpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}.pt"
        checkpoint_file = path.join(FSC_ENCODER_CHECKPOINT_DIR, checkpoint_filename)

        if not path.exists(FSC_ENCODER_CHECKPOINT_DIR):
            os.mkdir(FSC_ENCODER_CHECKPOINT_DIR)

        # Save the checkpoint
        torch.save({
            'epoch': self.epoch,
            'iter': self.iter,
            'elapsed_time': self._elapsed_time(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.last_loss
        }, checkpoint_file)

        # Create a link to the latest checkpoint (delete the old one if any)
        if path.exists(FSC_ENCODER_LATEST_CHECKPOINT):
            os.remove(FSC_ENCODER_LATEST_CHECKPOINT)
        os.symlink(checkpoint_file, path.abspath(FSC_ENCODER_LATEST_CHECKPOINT))

        print(f"CHK - Saved checkpoint: {checkpoint_filename}", flush=True)


    def _cleanup_checkpoints(self, keep_first: int):
        files = [os.path.join(FSC_ENCODER_CHECKPOINT_DIR, f) for f in os.listdir(FSC_ENCODER_CHECKPOINT_DIR)]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # Newest files first

        for f in files[keep_first:]:
            os.remove(f)
    

    def train(self):
        print(f"Running training, batch size: {BATCH_SIZE}...", flush=True)

        while True:
            print(f"-" * 96)
            print(f"EPOCH {self.epoch} (dim: {EPOCH_DIM})", flush=True)
            print(f"-" * 96)
            
            self._run_train_one_epoch()
            self._run_validation()
            self._save_checkpoint()
            self._cleanup_checkpoints(10)

            self.epoch += 1


if __name__ == "__main__":
    print(f"Pytorch CUDA Version is: {torch.version.cuda}", flush=True)

    model = FSC_Encoder()

    trainer = Trainer(model, load_latest_checkpoint=True)
    trainer.train()
