# ================================================================
# evaluate_real.py — Evaluate SuperPoint on REAL photographs (W × H)
# ================================================================

import os
import cv2
import torch
import numpy as np

from model import SuperPointDetector


# ================================================================
# Preprocess real image for model
# ================================================================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return gray, t


# ================================================================
# Extract predicted keypoints from probability map
# ================================================================
def extract_keypoints(prob_map, percentile=99.8):
    prob = prob_map[0, 0].cpu().numpy()
    thr = np.percentile(prob, percentile)     # auto-adaptive threshold
    ys, xs = np.where(prob > thr)
    return list(zip(xs, ys))


# ================================================================
# Save colored heatmap
# ================================================================
def save_heatmap(prob_map, filename):
    prob = prob_map[0, 0].cpu().numpy()
    prob_norm = (prob / (prob.max() + 1e-8) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
    cv2.imwrite(filename, heat)


# ================================================================
# Draw predicted keypoints on real image
# ================================================================
def draw_predictions(img, pred_kps, filename):
    vis = img.copy()
    for (x, y) in pred_kps:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imwrite(filename, vis)


# ================================================================
# MAIN — Evaluate model on real photographs
# ================================================================
def evaluate_real(W=None, H=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating REAL PHOTOS on:", device)

    os.makedirs("real_output", exist_ok=True)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model = SuperPointDetector().to(device)
    ckpt = torch.load("checkpoint_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------
    image_folder = "real_images"
    exts = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(exts)]

    if not image_files:
        print("No images found in real_images/")
        return

    print(f"Found {len(image_files)} real photos.")

    # ------------------------------------------------------------
    # Process each real photo
    # ------------------------------------------------------------
    for name in image_files:

        path = os.path.join(image_folder, name)
        img = cv2.imread(path)

        if img is None:
            print(f"Error loading image: {name}")
            continue

        # --------------------------------------------------------
        # Optional: resize image to user-specified W × H
        # --------------------------------------------------------
        if W is not None and H is not None:
            img = cv2.resize(img, (W, H))

        # Preprocess
        img_gray, img_t = preprocess(img)
        img_t = img_t.to(device)

        with torch.no_grad():
            logits, prob_map = model(img_t)

        pred_kps = extract_keypoints(prob_map)

        # --------------------------------------------------------
        # Save results
        # --------------------------------------------------------
        base = os.path.splitext(name)[0]

        # grayscale
        cv2.imwrite(
            f"real_output/{base}_gray.png",
            (img_gray * 255).astype(np.uint8)
        )

        # heatmap
        save_heatmap(prob_map, f"real_output/{base}_heatmap.png")

        # predicted points
        draw_predictions(img, pred_kps, f"real_output/{base}_pred.png")

        print(f"✓ Saved outputs for {name}")

    print("\nAll results saved in real_output/")
    print("Done!")


# ================================================================
# Run
# ================================================================
if __name__ == "__main__":
    # default resolution = model input resolution
    evaluate_real(W=256, H=320)
