import os
import cv2
import numpy as np


# ===============================================================
# Utility helpers
# ===============================================================

def random_color(minv=30, maxv=225):
    """Generate a random RGB color."""
    return np.random.randint(minv, maxv + 1, 3).tolist()


def ensure_contrast(bg, fg, thr=40):
    """Ensure foreground color differs enough from background."""
    while abs(np.mean(bg) - np.mean(fg)) < thr:
        fg = random_color()
    return fg


def add_smoothed_noise(img, sigma=10):
    """Add Gaussian noise + blur for smoothness."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


def far_enough(pt, pts, d):
    """Check minimum distance to all points."""
    return all((pt[0] - x)**2 + (pt[1] - y)**2 >= d*d for x, y in pts)


# ===============================================================
# Shape generators
# ===============================================================

def random_triangle(W, H, d=20):
    pts = []
    while len(pts) < 3:
        x = np.random.randint(5, W - 5)
        y = np.random.randint(5, H - 5)
        if far_enough([x, y], pts, d):
            pts.append([x, y])
    return pts


# ===============================================================
# NEW AND SAFE QUADRILATERAL GENERATOR
# ===============================================================

def random_quadrilateral(W, H, jitter=25):
    """
    EASIEST & SAFEST VERSION:
    1. Create axis-aligned rectangle
    2. Add jitter to each corner
    Always convex, never self-intersecting.
    """
    # rectangle top-left
    x1 = np.random.randint(40, W - 120)
    y1 = np.random.randint(40, H - 120)

    # width/height
    w = np.random.randint(40, 120)
    h = np.random.randint(40, 120)

    # base rectangle corners
    corners = np.array([
        [x1,     y1    ],  # top-left
        [x1 + w, y1    ],  # top-right
        [x1 + w, y1+h  ],  # bottom-right
        [x1,     y1+h  ]   # bottom-left
    ], dtype=np.float32)

    # jitter each corner
    corners += np.random.randint(-jitter, jitter + 1, corners.shape)

    # clamp inside frame
    corners[:, 0] = np.clip(corners[:, 0], 5, W - 5)
    corners[:, 1] = np.clip(corners[:, 1], 5, H - 5)

    return corners.tolist()


def random_star(W, H, center_r=80, d=20):
    cx = np.random.randint(center_r, W - center_r)
    cy = np.random.randint(center_r, H - center_r)

    pts = [[cx, cy]]
    while len(pts) < 6:
        ox = np.clip(cx + np.random.randint(-center_r, center_r), 5, W - 5)
        oy = np.clip(cy + np.random.randint(-center_r, center_r), 5, H - 5)
        if far_enough([ox, oy], pts, d):
            pts.append([ox, oy])
    return pts


def random_chessboard(W, H, rows=3, cols=4):
    margin = 20
    cb_w = np.random.randint(100, min(200, W - 2*margin))
    cb_h = np.random.randint(100, min(200, H - 2*margin))

    x0 = np.random.randint(margin, W - margin - cb_w)
    y0 = np.random.randint(margin, H - margin - cb_h)

    cell_w = cb_w // cols
    cell_h = cb_h // rows

    keypoints = [
        [x0 + c*cell_w, y0 + r*cell_h]
        for r in range(rows + 1)
        for c in range(cols + 1)
    ]

    return keypoints, (x0, y0, cb_w, cb_h)


def random_cube(W, H):
    cx = np.random.randint(60, W - 60)
    cy = np.random.randint(60, H - 60)
    C = [cx, cy]

    inner = []
    base_angle = np.random.uniform(0, 2 * np.pi)

    for k in range(3):
        a = base_angle + k * 2 * np.pi / 3
        d = np.random.randint(30, 80)
        ix = np.clip(int(cx + d * np.cos(a)), 5, W - 5)
        iy = np.clip(int(cy + d * np.sin(a)), 5, H - 5)
        inner.append([ix, iy])

    I1, I2, I3 = inner

    def outer(A, B):
        mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
        dx, dy = mx - cx, my - cy
        scale = np.random.uniform(1.2, 1.8)
        ox = np.clip(int(cx + dx * scale), 5, W - 5)
        oy = np.clip(int(cy + dy * scale), 5, H - 5)
        return [ox, oy]

    O12 = outer(I1, I2)
    O23 = outer(I2, I3)
    O31 = outer(I3, I1)

    return [C, I1, I2, I3, O12, O23, O31]


# ===============================================================
# Drawing functions
# ===============================================================

def draw_triangle(img, pts, color):
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)


def draw_quadrilateral(img, pts, color):
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)


def draw_star(img, pts, color):
    C = pts[0]
    for p in pts[1:]:
        cv2.line(img, tuple(C), tuple(p), color, 3)


def draw_chessboard(img, x0, y0, cb_w, cb_h, rows=3, cols=4):
    cell_w = cb_w // cols
    cell_h = cb_h // rows

    c1 = random_color()
    c2 = ensure_contrast(c1, random_color())

    for r in range(rows):
        for c in range(cols):
            col = c1 if (r + c) % 2 == 0 else c2
            x1 = x0 + c * cell_w
            y1 = y0 + r * cell_h
            cv2.rectangle(img, (x1, y1), (x1 + cell_w, y1 + cell_h), col, -1)


def draw_cube(img, pts):
    C, I1, I2, I3, O12, O23, O31 = pts
    faces = [
        [C, I1, O12, I2],
        [C, I2, O23, I3],
        [C, I3, O31, I1]
    ]

    colors = [
        random_color(),
        ensure_contrast(random_color(), random_color()),
        ensure_contrast(random_color(), random_color())
    ]

    for face, col in zip(faces, colors):
        arr = np.array(face, np.int32)
        cv2.fillPoly(img, [arr], col)
        cv2.polylines(img, [arr], True, (0, 0, 0), 2)


# ===============================================================
# Homography augmentation
# ===============================================================

def random_homography(W, H, min_dist=100):
    cx, cy = W // 2, H // 2

    def rp(xmin, xmax, ymin, ymax):
        while True:
            x = np.random.randint(xmin, xmax)
            y = np.random.randint(ymin, ymax)
            if (x - cx)**2 + (y - cy)**2 >= min_dist**2:
                return [x, y]

    src = np.array([
        rp(0, cx, 0, cy),
        rp(cx, W, 0, cy),
        rp(cx, W, cy, H),
        rp(0, cx, cy, H)
    ], np.float32)

    dst = np.array([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]
    ], np.float32)

    return cv2.getPerspectiveTransform(src, dst)


def maybe_add_rotation(Hm, W, H):
    k = np.random.randint(0, 4)
    if k == 0:
        return Hm

    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, 90 * k, 1.0)
    R = np.vstack([M, [0, 0, 1]]).astype(np.float32)

    return R @ Hm


def apply_homography(img, keypoints, Hm, W, H):
    warped = cv2.warpPerspective(img, Hm, (W, H))

    if not keypoints:
        return warped, []

    pts = np.array(keypoints, np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), np.float32)])

    warped_h = (Hm @ pts_h.T).T
    xs = warped_h[:, 0] / warped_h[:, 2]
    ys = warped_h[:, 1] / warped_h[:, 2]

    transformed = [
        [int(x), int(y)]
        for x, y in zip(xs, ys)
        if 0 <= x < W and 0 <= y < H
    ]

    return warped, transformed


def augment_with_homography(img, kps, W, H):
    Hm = random_homography(W, H)
    Hm = maybe_add_rotation(Hm, W, H)
    return apply_homography(img, kps, Hm, W, H)


# ===============================================================
# Image generators
# ===============================================================

def generate_simple_shape(shape, W, H):
    bg = random_color()
    img = np.full((H, W, 3), bg, np.uint8)
    fg = ensure_contrast(bg, random_color())

    if shape == "triangle":
        kps = random_triangle(W, H)
        draw_triangle(img, kps, fg)

    elif shape == "quadrilateral":
        kps = random_quadrilateral(W, H)
        draw_quadrilateral(img, kps, fg)

    elif shape == "star":
        kps = random_star(W, H)
        draw_star(img, kps, fg)

    elif shape == "chessboard":
        kps, box = random_chessboard(W, H)
        draw_chessboard(img, *box)

    elif shape == "cube":
        kps = random_cube(W, H)
        draw_cube(img, kps)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    return add_smoothed_noise(img), kps


def generate_multiple_shapes(W, H, max_shapes=4):
    bg = random_color()
    img = np.full((H, W, 3), bg, np.uint8)

    mask = np.zeros((H, W), np.uint8)
    all_kps = []

    shape_types = ["triangle", "quadrilateral", "star"]

    attempts = 0
    max_attempts = 80

    while len(all_kps) < max_shapes * 6 and attempts < max_attempts:
        attempts += 1
        st = np.random.choice(shape_types)

        # generate consistent shapes
        if st == "triangle":
            pts = random_triangle(W, H)
        elif st == "quadrilateral":
            pts = random_quadrilateral(W, H)
        else:
            pts = random_star(W, H)

        # mask check
        tmp = np.zeros_like(mask)
        if st == "star":
            C = pts[0]
            for p in pts[1:]:
                cv2.line(tmp, tuple(C), tuple(p), 255, 3)
        else:
            cv2.fillPoly(tmp, [np.array(pts, np.int32)], 255)

        if np.any(mask & tmp):
            continue

        col = ensure_contrast(bg, random_color())
        if st == "triangle":
            draw_triangle(img, pts, col)
        elif st == "quadrilateral":
            draw_quadrilateral(img, pts, col)
        else:
            draw_star(img, pts, col)

        mask |= tmp
        all_kps.extend(pts)

    img = add_smoothed_noise(img)
    return img, all_kps


# ===============================================================
# Saving helpers
# ===============================================================

def save_keypoint_mask(kps, W, H, path):
    out = np.zeros((H, W, 3), np.uint8)
    for x, y in kps:
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imwrite(path, out)


def save_paired_vis(img, kps, folder, prefix, idx, W, H):
    mask = np.zeros((H, W, 3), np.uint8)
    for x, y in kps:
        cv2.circle(mask, (int(x), int(y)), 3, (0, 255, 0), -1)

    paired = np.concatenate([img, mask], axis=1)
    out_path = f"{folder}/{prefix}_{idx:03d}_pair.png"
    cv2.imwrite(out_path, paired)


def save_example(img, kps, folder, prefix, idx, W, H):
    base = f"{folder}/{prefix}_{idx:03d}"

    cv2.imwrite(base + ".png", img)
    np.save(base + ".npy", np.array(kps, np.int32))

    save_keypoint_mask(kps, W, H, base + "_mask.png")
    save_paired_vis(img, kps, folder, prefix, idx, W, H)


# ===============================================================
# Dataset generator
# ===============================================================

def generate_dataset(output_dir="dataset", n=30, W=320, H=240):
    os.makedirs(output_dir, exist_ok=True)

    simple = ["triangle", "quadrilateral", "star"]
    complex_shapes = ["chessboard", "cube"]
    groups = simple + complex_shapes + ["multi"]

    for g in groups:
        os.makedirs(f"{output_dir}/{g}/original", exist_ok=True)
        os.makedirs(f"{output_dir}/{g}/homography", exist_ok=True)

    for s in simple:
        for i in range(n):
            img, kps = generate_simple_shape(s, W, H)
            save_example(img, kps, f"{output_dir}/{s}/original", s, i, W, H)
            h_img, h_kps = augment_with_homography(img, kps, W, H)
            save_example(h_img, h_kps, f"{output_dir}/{s}/homography", s, i, W, H)

    for s in complex_shapes:
        for i in range(n):
            img, kps = generate_simple_shape(s, W, H)
            save_example(img, kps, f"{output_dir}/{s}/original", s, i, W, H)
            h_img, h_kps = augment_with_homography(img, kps, W, H)
            save_example(h_img, h_kps, f"{output_dir}/{s}/homography", s, i, W, H)

    for i in range(n):
        img, kps = generate_multiple_shapes(W, H, max_shapes=4)
        save_example(img, kps, f"{output_dir}/multi/original", "multi", i, W, H)
        h_img, h_kps = augment_with_homography(img, kps, W, H)
        save_example(h_img, h_kps, f"{output_dir}/multi/homography", "multi", i, W, H)

    print(f"\nDataset generation complete → saved to: {output_dir}")


# ===============================================================
# Entry point
# ===============================================================

if __name__ == "__main__":
    generate_dataset(output_dir="dataset", n=50, W=320, H=240)
