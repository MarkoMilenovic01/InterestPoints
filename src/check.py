# ================================================================
# check.py — sanity checks for SuperPoint implementation
# ================================================================

import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SuperPointDetector, LogitsToProbMap
from train import keypoints_to_grid, SyntheticDataset, loss_fn


# ================================================================
# 1. Shape check: random input through the model
# ================================================================

def test_model_shapes(H=256, W=256):
    print("\n[TEST] Model shape check")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperPointDetector().to(device)
    model.eval()

    x = torch.randn(1, 1, H, W).to(device)

    with torch.no_grad():
        logits, prob_map = model(x)

    print(" Input shape : ", x.shape)
    print(" Logits shape: ", logits.shape)
    print(" Prob map    : ", prob_map.shape)
    assert logits.shape[2] == H // 8 and logits.shape[3] == W // 8
    assert prob_map.shape[2] == H and prob_map.shape[3] == W
    print(" -> OK\n")


# ================================================================
# 2. Consistency: mask → grid → logits → decoder → back to pixel
# ================================================================

def test_mask_grid_decoder_consistency(size=256):
    print("\n[TEST] Mask–Grid–Decoder consistency")

    H = W = size
    Hc, Wc = H // 8, W // 8

    # 1) Create an empty mask with a single keypoint
    mask = np.zeros((H, W), dtype=np.float32)

    # Choose some random coarse cell and sub-pixel inside it
    cell_y, cell_x = 5, 7   # e.g. 5th row, 7th column of coarse grid
    sub_y,  sub_x  = 2, 6   # position inside the 8×8 cell

    y = cell_y * 8 + sub_y
    x = cell_x * 8 + sub_x
    mask[y, x] = 1.0

    print(f" Original keypoint at pixel: (y={y}, x={x})")

    # 2) Convert mask → 65-channel grid using your training code
    grid = keypoints_to_grid(mask)  # (Hc, Wc, 65)

    # Find which class is active in this (cell_y, cell_x)
    one_hot = grid[cell_y, cell_x]          # (65,)
    class_idx = np.argmax(one_hot)          # should be in [0..63]
    print(" Active class index in that cell:", class_idx)
    if class_idx == 64:
        print(" WARNING: Dustbin is active where a keypoint is present!")

    # 3) Build logits that strongly prefer that class in that single cell
    logits = np.zeros((1, 65, Hc, Wc), dtype=np.float32)
    logits[0, class_idx, cell_y, cell_x] = 20.0  # large positive logit

    logits_t = torch.from_numpy(logits).float()

    # 4) Decode back to full-resolution prob map
    decoder = LogitsToProbMap()
    with torch.no_grad():
        prob_map = decoder(logits_t)  # (1,1,H,W)

    prob = prob_map[0, 0].numpy()      # (H,W)
    flat_idx = np.argmax(prob)
    py = flat_idx // W
    px = flat_idx % W

    print(f" Decoded max prob at pixel: (y={py}, x={px})")

    if (py, px) == (y, x):
        print(" -> Perfect match. Consistency OK.\n")
    else:
        print(" -> MISMATCH! Something is off in mask/grid/decoder.\n")


# ================================================================
# 3. Dataset sanity: one sample shapes + one-hot checks
# ================================================================

def test_dataset_sample():
    print("\n[TEST] SyntheticDataset sample")

    ds = SyntheticDataset(size=256, homography_prob=0.7, mode="mixed")
    img, grid = ds[0]   # img: (1,H,W), grid: (65,Hc,Wc)

    print(" Image shape:", img.shape)
    print(" Grid shape :", grid.shape)

    # Check one-hot: sum over channels should be ~1 per cell
    cell_sums = grid.sum(dim=0)  # (Hc,Wc)
    min_sum = cell_sums.min().item()
    max_sum = cell_sums.max().item()
    print(" Cell-sum min / max:", min_sum, "/", max_sum)

    if abs(min_sum - 1.0) < 1e-5 and abs(max_sum - 1.0) < 1e-5:
        print(" -> All cells are proper one-hot (sum=1). OK.\n")
    else:
        print(" -> WARNING: some cells are not one-hot.\n")


# ================================================================
# 4. Tiny overfit: does loss go down on a small subset?
# ================================================================

def tiny_overfit(max_iters=200, batch_size=8):
    print("\n[TEST] Tiny overfit")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(" Running tiny overfit on:", device)

    # Small synthetic dataset (no homography, only simple shapes)
    ds = SyntheticDataset(size=256, homography_prob=0.0, mode="simple")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SuperPointDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    model.train()
    losses = []
    it = 0

    while it < max_iters:
        for img, target in loader:
            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if it % 20 == 0:
                print(f" [Tiny it {it:4d}] loss = {loss.item():.4f}")

            it += 1
            if it >= max_iters:
                break

    print(" Tiny overfit finished.")
    print("  Initial loss approx:", losses[0])
    print("  Min loss achieved  :", min(losses))
    print("  Final loss         :", losses[-1], "\n")


# ================================================================
# 5. Visualize one example (using best checkpoint if available)
# ================================================================

def visualize_one_example(checkpoint_path="checkpoint_best.pth"):
    print("\n[TEST] Visualize one example")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) One synthetic sample
    ds = SyntheticDataset(size=256, homography_prob=0.0, mode="simple")
    img, target = ds[0]  # img: (1,256,256)

    # 2) Build model
    model = SuperPointDetector().to(device)

    # Optionally load trained weights
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f" Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        print(" No checkpoint found -> using random weights (visual only).")

    model.eval()

    with torch.no_grad():
        img_batch = img.unsqueeze(0).to(device)  # (1,1,256,256)
        logits, prob_map = model(img_batch)

    prob = prob_map[0, 0].cpu().numpy()

    # Simple thresholding for predicted keypoints
    thr = np.percentile(prob, 99.5)   # or 99.8
    ys, xs = np.where(prob > thr)
    print(f" Predicted keypoints above threshold: {len(xs)}")

    # Convert image to BGR for drawing
    img_vis = (img[0].numpy() * 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

    # Draw predicted points in GREEN
    for x, y in zip(xs, ys):
        cv2.circle(img_vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    out_path = "debug_keypoints.png"
    cv2.imwrite(out_path, img_vis)
    print(f" Saved visualization to {out_path}\n")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    # # 1) pure shape sanity
    test_model_shapes(H=256, W=256)

    # # 2) label / decoder consistency
    test_mask_grid_decoder_consistency(size=256)

    # # 3) dataset sanity
    test_dataset_sample()

    # # 4) tiny overfit (quick check that loss goes down)
    tiny_overfit(max_iters=1000, batch_size=8)

    # 5) optional: visualize one example (after some training)
    #    this will still run even with random weights; you'll just see random dots
    visualize_one_example(checkpoint_path="checkpoint_best.pth")
