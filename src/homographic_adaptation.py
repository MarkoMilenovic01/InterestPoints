# ================================================================
# evaluate_real_ha.py — SuperPoint on REAL images + Homography Adaptation
# Input  folder: real_images/
# Output folder: homography_adaptation_outputs/
# ================================================================

import os
import glob
import cv2
import numpy as np
import torch

from model import SuperPointDetector
from generate_dataset import random_homography, maybe_add_rotation


INPUT_DIR = "real_images"
OUTPUT_DIR = "homography_adaptation_outputs"
CHECKPOINT_PATH = "checkpoint_best.pth"

NUM_H = 99
INCLUDE_IDENTITY = True

USE_PERCENTILE = 99.5
NMS_RADIUS = 1
MAX_KP = 500


def bgr_to_gray01(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def gray01_to_tensor(gray01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(gray01).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def infer_prob_map(model: SuperPointDetector, gray01: np.ndarray, device: str) -> np.ndarray:
    x = gray01_to_tensor(gray01).to(device)
    _, prob = model(x)
    return prob[0, 0].detach().cpu().numpy().astype(np.float32)


def warp_gray(gray01: np.ndarray, Hm: np.ndarray) -> np.ndarray:
    H, W = gray01.shape
    return cv2.warpPerspective(
        gray01,
        Hm,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)


def warp_back_map(map_hw: np.ndarray, Hinv: np.ndarray) -> np.ndarray:
    H, W = map_hw.shape
    return cv2.warpPerspective(
        map_hw,
        Hinv,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)


def warp_back_mask(mask_hw: np.ndarray, Hinv: np.ndarray) -> np.ndarray:
    H, W = mask_hw.shape
    return cv2.warpPerspective(
        mask_hw,
        Hinv,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)


def homography_adaptation(model, gray01, device, num_h=99, include_identity=True):
    H, W = gray01.shape
    acc_prob = np.zeros((H, W), dtype=np.float32)
    acc_mask = np.zeros((H, W), dtype=np.float32)
    ones = np.ones((H, W), dtype=np.float32)

    if include_identity:
        p0 = infer_prob_map(model, gray01, device)
        acc_prob += p0
        acc_mask += ones

    for _ in range(num_h):
        Hm = random_homography(W, H)
        Hm = maybe_add_rotation(Hm, W, H).astype(np.float32)

        det = float(np.linalg.det(Hm))
        if abs(det) < 1e-8:
            continue
        Hinv = np.linalg.inv(Hm).astype(np.float32)

        warped = warp_gray(gray01, Hm)
        p_warp = infer_prob_map(model, warped, device)

        acc_prob += warp_back_map(p_warp, Hinv)
        acc_mask += warp_back_mask(ones, Hinv)

    return (acc_prob / np.maximum(acc_mask, 1e-6)).astype(np.float32)


def extract_keypoints_nms(prob_hw, use_percentile=99.5, nms_radius=4, max_kp=500):
    thr = float(np.percentile(prob_hw, use_percentile))
    mask = prob_hw > thr
    if mask.sum() == 0:
        return []

    k = 2 * nms_radius + 1
    local_max = (prob_hw == cv2.dilate(prob_hw, np.ones((k, k), np.uint8)))
    keep = mask & local_max

    ys, xs = np.where(keep)
    if len(xs) == 0:
        return []

    scores = prob_hw[ys, xs]
    order = np.argsort(-scores)[:max_kp]
    return [(int(xs[i]), int(ys[i])) for i in order]


def save_heatmap_from_prob(prob_hw: np.ndarray, filename: str,
                           clip_percentile: float = 99.5,
                           gamma: float = 0.9):
    """
    clip_percentile: lower -> brighter overall but more saturation
    gamma (<1): brighter mid intensities (0.5 = sqrt)
    """
    prob = prob_hw.astype(np.float32)

    # Clip scale to percentile so the map isn't mostly dark
    scale = np.percentile(prob, clip_percentile)
    scale = max(float(scale), 1e-8)

    prob_norm = np.clip(prob / scale, 0.0, 1.0)

    # Gamma boost (brighten)
    prob_norm = prob_norm ** gamma

    prob_u8 = (prob_norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    cv2.imwrite(filename, heat)



def draw_pred_on_bgr(img_bgr, pred_kps, filename):
    vis = img_bgr.copy()
    for (x, y) in pred_kps:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imwrite(filename, vis)


def evaluate_real_ha(W=None, H=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Evaluating REAL PHOTOS + HA on:", device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        print("No real_images/ folder found.")
        return
    if not os.path.exists(CHECKPOINT_PATH):
        print("checkpoint_best.pth not found.")
        return

    # load model
    model = SuperPointDetector().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # images
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(INPUT_DIR, e)))
    paths = sorted(paths)

    if not paths:
        print("No images found in real_images/")
        return

    for p in paths:
        name = os.path.basename(p)
        base = os.path.splitext(name)[0]

        img = cv2.imread(p)
        if img is None:
            continue

        # resize FIRST (like evaluate_real.py)
        if W is not None and H is not None:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        gray01 = bgr_to_gray01(img)

        prob_single = infer_prob_map(model, gray01, device)
        prob_ha = homography_adaptation(model, gray01, device, num_h=NUM_H, include_identity=INCLUDE_IDENTITY)

        pred_kps = extract_keypoints_nms(prob_ha, use_percentile=USE_PERCENTILE, nms_radius=NMS_RADIUS, max_kp=MAX_KP)

        out_dir = os.path.join(OUTPUT_DIR, base)
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_dir, "input.png"), img)
        save_heatmap_from_prob(prob_single, os.path.join(out_dir, "heat_single.png"))
        save_heatmap_from_prob(prob_ha, os.path.join(out_dir, "heat_HA.png"))
        draw_pred_on_bgr(img, pred_kps, os.path.join(out_dir, "pred_HA.png"))

        print(f"✓ {name}  (pred={len(pred_kps)})")

    print("\nAll results saved in homography_adaptation_outputs/")


if __name__ == "__main__":
    evaluate_real_ha(W=256, H=320)  # choose multiples of 8
