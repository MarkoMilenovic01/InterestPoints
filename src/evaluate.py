# ================================================================
# evaluate.py — Evaluate SuperPoint on synthetic samples (W × H)
# ================================================================

import os
import torch
import numpy as np
import cv2

from model import SuperPointDetector
from generate_dataset import (
    generate_simple_shape,
    generate_multiple_shapes
)


# ================================================================
# Preprocessing
# ================================================================
def preprocess(img):
    """Convert BGR image → normalized grayscale tensor."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0)
    return img_gray, img_t


# ================================================================
# Save heatmap
# ================================================================
def save_heatmap(prob_map, filename):
    prob = prob_map[0, 0].cpu().numpy()
    prob_norm = (prob / prob.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
    cv2.imwrite(filename, heat)


# ================================================================
# Draw GT (red X) and predicted (green dot)
# ================================================================
def draw_merge(img_gray, gt_kps, pred_kps, filename):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Ground truth keypoints (red X)
    for (x, y) in gt_kps:
        cv2.drawMarker(
            vis, (int(x), int(y)),
            (0, 0, 255),   # red
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=8,
            thickness=2
        )

    # Predicted keypoints (green dots)
    for (x, y) in pred_kps:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imwrite(filename, vis)


# ================================================================
# SuperPoint-style NMS keypoint extraction
# ================================================================
def get_predicted_points(prob_map, threshold=0.015, nms_dist=4):
    """
    Extract keypoints from probability map using:
    1) thresholding
    2) 3x3 or nxn local maximum suppression (default 4x4)
    """
    prob = prob_map[0, 0].cpu().numpy()

    # Step 1: threshold
    mask = prob > threshold
    if mask.sum() == 0:
        return []

    # Step 2: Non-Maximum Suppression (dilated max filter)
    kernel = np.ones((nms_dist, nms_dist), np.uint8)
    local_max = (prob == cv2.dilate(prob, kernel))

    final_mask = mask & local_max
    ys, xs = np.where(final_mask)

    # Return list of (x, y)
    return list(zip(xs, ys))


# ================================================================
# MAIN EVALUATION
# ================================================================
def evaluate(W=320, H=240):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating on:", device)
    print(f"Image resolution = {W} × {H}")

    os.makedirs("eval_output", exist_ok=True)

    # ------------------------------------------------------------
    # Load best model checkpoint
    # ------------------------------------------------------------
    model = SuperPointDetector().to(device)

    if not os.path.exists("checkpoint_best.pth"):
        raise FileNotFoundError("checkpoint_best.pth not found.")

    ckpt = torch.load("checkpoint_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ------------------------------------------------------------
    # Generate & evaluate 10 synthetic samples
    # ------------------------------------------------------------
    for i in range(10):
        mode = np.random.choice(["simple", "complex", "multi"], p=[0.4, 0.4, 0.2])

        if mode == "simple":
            shape = np.random.choice(["triangle", "quadrilateral", "star"])
            img, gt_kps = generate_simple_shape(shape, W=W, H=H)

        elif mode == "complex":
            shape = np.random.choice(["chessboard", "cube"])
            img, gt_kps = generate_simple_shape(shape, W=W, H=H)

        else:
            img, gt_kps = generate_multiple_shapes(W=W, H=H, max_shapes=4)

        # --------------------------------------------------------
        # Preprocess
        # --------------------------------------------------------
        img_gray, img_t = preprocess(img)
        img_t = img_t.to(device)

        # --------------------------------------------------------
        # Forward pass
        # --------------------------------------------------------
        with torch.no_grad():
            logits, prob_map = model(img_t)

        # --------------------------------------------------------
        # Extract keypoints
        # --------------------------------------------------------
        pred_kps = get_predicted_points(prob_map)

        # --------------------------------------------------------
        # Save visual outputs
        # --------------------------------------------------------
        cv2.imwrite(f"eval_output/sample_{i}_input.png", img)
        save_heatmap(prob_map, f"eval_output/sample_{i}_heatmap.png")

        # Predicted only
        pred_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        for (x, y) in pred_kps:
            cv2.circle(pred_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.imwrite(f"eval_output/sample_{i}_pred.png", pred_vis)

        # GT + Pred (merged)
        draw_merge(
            img_gray,
            gt_kps=gt_kps,
            pred_kps=pred_kps,
            filename=f"eval_output/sample_{i}_merge.png"
        )

        print(f"[✓] Saved eval sample {i}")

    print("\nAll results saved to: eval_output/\n")


# ================================================================
# Run evaluation
# ================================================================
if __name__ == "__main__":
    evaluate(W=320, H=240)
