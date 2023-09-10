import sys
import torch

import torchvision
torchvision.disable_beta_transforms_warning()

from dataset import FSC_Dataset
from model import FSC_Encoder
import matplotlib.pyplot as plt


def fill_axis(ax, results):
    rows = [list(x) for x in zip(*results)]

    num_cols = len(rows[0])

    img = torchvision.utils.make_grid(rows[0] + rows[1], nrow=num_cols, padding=0)
    ax.imshow(img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1.0)
    ax.axis('off')

    for i in range(num_cols):
        ax.add_patch(plt.Rectangle((32 * i, 0), 32, 32, fill=False, edgecolor='gray', linewidth=1))
        ax.add_patch(plt.Rectangle((32 * i, 32), 32, 32, fill=False, edgecolor='gray', linewidth=1))
        ax.margins(0.001)
        ax.text(32 * i + 16, 32, f"{rows[2][i]:.3f}", ha='center', va='center', c='blue', backgroundcolor='white', fontsize=24)


def run_inference(model: FSC_Encoder):
    dataset = FSC_Dataset(
        "/home/rutayisire/unimore/cv/md-scanner/fsc-dataset/dataset.csv",
        "/home/rutayisire/unimore/cv/md-scanner/fsc-dataset"
        )
    
    font = dataset.pick_random_font()

    print(f"Random font: {font}")

    _, axs = plt.subplots(2, 1, figsize=(15, 8))

    sf_results = []
    df_results = []

    for _ in range(10):
        x1, x2, _ = dataset.pick_same_font_input(font)

        y1 = model(torch.unsqueeze(x1, 0))
        y2 = model(torch.unsqueeze(x2, 0))
        dist = torch.norm(y1 - y2)

        sf_results.append((x1, x2, dist))
    
    for _ in range(10):
        x1, x2, _ = dataset.pick_diff_font_input(font)

        y1 = model(torch.unsqueeze(x1, 0))
        y2 = model(torch.unsqueeze(x2, 0))
        dist = torch.norm(y1 - y2)

        df_results.append((x1, x2, dist))

    fill_axis(axs[0], sf_results)
    fill_axis(axs[1], df_results)

    plt.show()


if __name__ == "__main__":
    checkpoint_file = sys.argv[1]

    print(f"Loading checkpoint \"{checkpoint_file}\"...")
    checkpoint = torch.load(checkpoint_file)

    print(f"Loading model and setting checkpoint up...")
    model = FSC_Encoder()
    model.load_checkpoint(checkpoint)
    model.eval()

    while True:
        run_inference(model)

