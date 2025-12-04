# ================================================================
# train.py — SuperPoint training with synthetic data (50k iters)
# ================================================================

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from model import SuperPointDetector
from generate_dataset import (
    generate_simple_shape,
    generate_multiple_shapes,
    augment_with_homography
)


# ================================================================
# Checkpoint utilities
# ================================================================

def save_checkpoint(model, optimizer, iteration, path="checkpoint.pth"):
    checkpoint = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"[✓] Saved checkpoint at {iteration} → {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if not os.path.exists(path):
        print("No checkpoint found → starting fresh.")
        return 0

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    it = checkpoint["iteration"]

    print(f"[✓] Loaded checkpoint (iteration={it})")
    return it


# ================================================================
# Best-loss checkpoint
# ================================================================

def save_best_checkpoint(model, optimizer, iteration, best_loss, path="checkpoint_best.pth"):
    checkpoint = {
        "iteration": iteration,
        "best_loss": best_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"[✓] NEW BEST at iteration {iteration} (loss={best_loss:.6f})")


def load_best_checkpoint(path="checkpoint_best.pth"):
    if not os.path.exists(path):
        print("No best-loss checkpoint.")
        return None, float("inf")

    ckpt = torch.load(path, map_location="cpu")
    print(f"[✓] Loaded best-loss checkpoint (iter={ckpt['iteration']}, loss={ckpt['best_loss']:.6f})")
    return ckpt, ckpt["best_loss"]


# ================================================================
# Mask → 65-channel grid
# ================================================================

def keypoints_to_grid(mask):
    H, W = mask.shape
    Hc, Wc = H // 8, W // 8

    m = mask.reshape(Hc, 8, Wc, 8)
    m = np.transpose(m, (0, 2, 1, 3))
    m = m.reshape(Hc, Wc, 64)

    grid = np.zeros((Hc, Wc, 65), dtype=np.float32)
    grid[..., 64] = 1.0

    for i in range(Hc):
        for j in range(Wc):
            idxs = np.where(m[i, j] > 0)[0]
            if len(idxs) > 0:
                keep = np.random.choice(idxs)
                grid[i, j, :] = 0
                grid[i, j, keep] = 1.0

    return grid


# ================================================================
# Synthetic Dataset
# ================================================================

class SyntheticDataset(Dataset):
    def __init__(self, width=320, height=240, homography_prob=0.5, mode="mixed"):
        self.W = width
        self.H = height
        self.h_prob = homography_prob
        self.mode = mode

        self.simple = ["triangle", "quadrilateral", "star"]
        self.complex = ["chessboard", "cube"]

    def __len__(self):
        return 9999999  # infinite

    def __getitem__(self, idx):

        # -------- shape selection --------
        if self.mode == "simple":
            shape = np.random.choice(self.simple)
            img, kps = generate_simple_shape(shape, self.W, self.H)

        elif self.mode == "complex":
            shape = np.random.choice(self.complex)
            img, kps = generate_simple_shape(shape, self.W, self.H)

        elif self.mode == "multi":
            img, kps = generate_multiple_shapes(self.W, self.H, max_shapes=4)

        else:
            mode = np.random.choice(["simple", "complex", "multi"], p=[0.4, 0.4, 0.2])

            if mode == "simple":
                shape = np.random.choice(self.simple)
                img, kps = generate_simple_shape(shape, self.W, self.H)

            elif mode == "complex":
                shape = np.random.choice(self.complex)
                img, kps = generate_simple_shape(shape, self.W, self.H)

            else:
                img, kps = generate_multiple_shapes(self.W, self.H, max_shapes=4)

        # -------- homography augmentation --------
        if np.random.rand() < self.h_prob:
            img, kps = augment_with_homography(img, kps, W=self.W, H=self.H)

        # -------- grayscale --------
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # -------- keypoint mask --------
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        for (x, y) in kps:
            if 0 <= x < self.W and 0 <= y < self.H:
                mask[int(y), int(x)] = 1.0

        grid = keypoints_to_grid(mask)

        img_t = torch.from_numpy(img_gray).unsqueeze(0)
        grid_t = torch.from_numpy(grid).permute(2, 0, 1).float()

        return img_t, grid_t

# ================================================================
# Loss
# ================================================================

def loss_fn(logits, target):
    idx = torch.argmax(target, dim=1)
    return F.cross_entropy(logits, idx)


# ================================================================
# TRAINING — 50,000 ITERATIONS + ReduceLROnPlateau
# ================================================================

def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    dataset = SyntheticDataset(width=320, height=240, homography_prob=0.7, mode="mixed")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    model = SuperPointDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # #  Reduce LR when plateau
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.5,
    #     patience=5000,      # if loss does not improve for 1000 iterations → reduce LR
    #     cooldown=200,
    #     min_lr=1e-6,
    # )

    start_iter = load_checkpoint(model, optimizer)
    _, best_loss = load_best_checkpoint()

    loss_history = []

    max_iters = 3000
    iteration = start_iter

    print(f"Starting at iteration {iteration}")
    print(f"Best loss so far = {best_loss:.6f}")

    model.train()

    while iteration < max_iters:
        for img, target in loader:

            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())


            if iteration % 200 == 0:
                print(f"[{iteration}] loss={loss.item():.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

            if iteration % 2000 == 0 and iteration != start_iter:
                save_checkpoint(model, optimizer, iteration)
                pd.DataFrame({"loss": loss_history}).to_csv("loss_history.csv", index=False)

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_best_checkpoint(model, optimizer, iteration, best_loss)

            iteration += 1
            if iteration >= max_iters:
                break

    save_checkpoint(model, optimizer, iteration, "checkpoint_final.pth")
    pd.DataFrame({"loss": loss_history}).to_csv("loss_history.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss — 50,000 Iterations")
    plt.savefig("training_loss.png")
    plt.close()

    print("Training finished.")
    print(f"Best final loss = {best_loss:.6f}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    train()
