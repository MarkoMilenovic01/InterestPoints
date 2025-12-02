import os
import cv2
import numpy as np

# ===============================================================
# Utility helpers
# ===============================================================

def random_color(minv: int = 30, maxv: int = 225):
    """Random RGB color."""
    return np.random.randint(minv, maxv + 1, 3).tolist()


def ensure_contrast(bg, fg, thr: float = 40):
    """Ensure fg is visually different from bg."""
    while abs(np.mean(bg) - np.mean(fg)) < thr:
        fg = random_color()
    return fg


def add_smoothed_noise(img, sigma: float = 10):
    """Add Gaussian noise and blur a bit."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


def far_enough(pt, pts, d: float):
    """Check that pt is at least distance d from all pts."""
    return all((pt[0] - x) ** 2 + (pt[1] - y) ** 2 >= d * d for x, y in pts)


# ===============================================================
# Shape generators
# ===============================================================

def random_triangle(w: int, h: int, d: int = 20):
    pts = []
    while len(pts) < 3:
        x = int(np.random.randint(5, w - 5))
        y = int(np.random.randint(5, h - 5))
        if far_enough([x, y], pts, d):
            pts.append([x, y])
    return pts


def random_quadrilateral(w: int, h: int, d: int = 20):
    pts = []
    while len(pts) < 4:
        x = int(np.random.randint(5, w - 5))
        y = int(np.random.randint(5, h - 5))
        if far_enough([x, y], pts, d):
            pts.append([x, y])
    return pts


def random_star(w: int, h: int, center_r: int = 100, d: int = 20):
    cx = int(np.random.randint(center_r, w - center_r))
    cy = int(np.random.randint(center_r, h - center_r))
    pts = [[cx, cy]]  # center

    while len(pts) < 6:
        ox = int(np.clip(cx + np.random.randint(-center_r, center_r), 5, w - 5))
        oy = int(np.clip(cy + np.random.randint(-center_r, center_r), 5, h - 5))
        if far_enough([ox, oy], pts, d):
            pts.append([ox, oy])

    return pts


def random_chessboard(w: int, h: int, rows: int = 3, cols: int = 4):
    margin = 20
    x0 = int(np.random.randint(margin, w - margin - 200))
    y0 = int(np.random.randint(margin, h - margin - 200))

    cb_w = int(np.random.randint(120, 200))
    cb_h = int(np.random.randint(120, 200))

    cell_w = cb_w // cols
    cell_h = cb_h // rows

    keypoints = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            x = int(x0 + c * cell_w)
            y = int(y0 + r * cell_h)
            keypoints.append([x, y])

    return keypoints, (x0, y0, cb_w, cb_h)


def random_cube(w: int, h: int):
    cx = int(np.random.randint(60, w - 60))
    cy = int(np.random.randint(60, h - 60))
    C = [cx, cy]

    inner = []
    base = np.random.uniform(0, 2 * np.pi)

    # inner triangle
    for k in range(3):
        a = base + k * 2 * np.pi / 3
        d = int(np.random.randint(30, 100))
        ix = int(cx + d * np.cos(a))
        iy = int(cy + d * np.sin(a))
        inner.append([int(np.clip(ix, 5, w - 5)), int(np.clip(iy, 5, h - 5))])

    I1, I2, I3 = inner

    def outer(A, B):
        mx = (A[0] + B[0]) / 2
        my = (A[1] + B[1]) / 2
        dx, dy = mx - cx, my - cy
        scale = np.random.uniform(1.2, 1.8)
        ox = int(cx + dx * scale)
        oy = int(cy + dy * scale)
        return [int(np.clip(ox, 5, w - 5)), int(np.clip(oy, 5, h - 5))]

    O12 = outer(I1, I2)
    O23 = outer(I2, I3)
    O31 = outer(I3, I1)

    return [C, I1, I2, I3, O12, O23, O31]


# ===============================================================
# Drawing functions
# ===============================================================

def draw_triangle(img, pts, color):
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)


def draw_quadrilateral(img, pts, color):
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)


def draw_star(img, pts, color):
    C = pts[0]
    for p in pts[1:]:
        cv2.line(img, tuple(C), tuple(p), color, 3)


def draw_chessboard(img, x0, y0, cb_w, cb_h, rows: int = 3, cols: int = 4):
    cell_w, cell_h = cb_w // cols, cb_h // rows
    c1 = random_color()
    c2 = ensure_contrast(c1, random_color())
    for r in range(rows):
        for c in range(cols):
            color = c1 if (r + c) % 2 == 0 else c2
            x1, y1 = x0 + c * cell_w, y0 + r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)


def draw_cube(img, pts):
    C, I1, I2, I3, O12, O23, O31 = pts
    faces = [
        [C, I1, O12, I2],
        [C, I2, O23, I3],
        [C, I3, O31, I1],
    ]

    f1 = random_color()
    f2 = ensure_contrast(f1, random_color())
    f3 = ensure_contrast(f2, random_color())
    colors = [f1, f2, f3]

    for face, fc in zip(faces, colors):
        face_arr = np.array(face, dtype=np.int32)
        cv2.fillPoly(img, [face_arr], fc)
        cv2.polylines(img, [face_arr], True, (0, 0, 0), 2)


# ===============================================================
# Homography helpers
# ===============================================================

def random_homography(size: int = 256, min_dist: int = 40):
    """
    Generate a homography by picking 4 points (one per quadrant),
    each sufficiently far from the image center, and mapping them
    to the 4 image corners.
    """
    w = h = size
    cx, cy = w // 2, h // 2

    def random_point(xmin, xmax, ymin, ymax):
        while True:
            x = np.random.randint(xmin, xmax)
            y = np.random.randint(ymin, ymax)
            if (x - cx) ** 2 + (y - cy) ** 2 >= min_dist ** 2:
                return [x, y]

    # source points: TL, TR, BR, BL
    src = np.array([
        random_point(0, cx, 0, cy),      # top-left
        random_point(cx, w, 0, cy),      # top-right
        random_point(cx, w, cy, h),      # bottom-right
        random_point(0, cx, cy, h),      # bottom-left
    ], dtype=np.float32)

    # destination points: exact image corners
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    return H


def maybe_add_rotation(H, size: int = 256):
    """
    Optionally compose H with a random 0/90/180/270 degree rotation.
    """
    k = np.random.randint(0, 4)  # 0,1,2,3 -> 0, 90, 180, 270 degrees
    if k == 0:
        return H

    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, 90 * k, 1.0)  # 2x3
    R = np.vstack([M, [0, 0, 1]]).astype(np.float32)  # -> 3x3
    return R @ H


def apply_homography(img, keypoints, H, size: int = 256):
    """
    Apply homography H to image and keypoints.
    Remove keypoints that fall outside the image.
    """
    warped_img = cv2.warpPerspective(img, H, (size, size))

    if not keypoints:
        return warped_img, []

    pts = np.array(keypoints, dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])  # Nx3

    warped_h = (H @ pts_h.T).T  # Nx3

    # normalize
    xs = warped_h[:, 0] / warped_h[:, 2]
    ys = warped_h[:, 1] / warped_h[:, 2]

    transformed = []
    for x, y in zip(xs, ys):
        if 0 <= x < size and 0 <= y < size:
            transformed.append([int(x), int(y)])

    return warped_img, transformed


def augment_with_homography(img, keypoints, size: int = 256):
    """Wrapper: generate H, possibly rotate, apply to image & keypoints."""
    H = random_homography(size=size)
    H = maybe_add_rotation(H, size=size)
    return apply_homography(img, keypoints, H, size=size)


# ===============================================================
# Image generators
# ===============================================================

def generate_simple_shape(shape: str, size: int = 256):
    """Generate a single shape image + keypoints."""
    assert size % 8 == 0

    bg = random_color()
    img = np.full((size, size, 3), bg, np.uint8)
    fg = ensure_contrast(bg, random_color())

    if shape == "triangle":
        kps = random_triangle(size, size)
        draw_triangle(img, kps, fg)

    elif shape == "quadrilateral":
        kps = random_quadrilateral(size, size)
        draw_quadrilateral(img, kps, fg)

    elif shape == "star":
        kps = random_star(size, size)
        draw_star(img, kps, fg)

    elif shape == "chessboard":
        kps, (x0, y0, w, h) = random_chessboard(size, size)
        draw_chessboard(img, x0, y0, w, h)

    elif shape == "cube":
        kps = random_cube(size, size)
        draw_cube(img, kps)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    img = add_smoothed_noise(img)
    return img, kps


def generate_multiple_shapes(size: int = 256, max_shapes: int = 4):
    """
    Generate an image with several NON-OVERLAPPING simple shapes:
    triangle, quadrilateral, star.

    pick shape
    generate pts
    draw tmp mask
    if overlap: skip
    draw shape
    update mask, keypoints

    """
    assert size % 8 == 0

    bg  = random_color()
    img = np.full((size, size, 3), bg, np.uint8)
    mask = np.zeros((size, size), np.uint8)

    all_kps = []
    shape_types = ["triangle", "quadrilateral", "star"]
    attempts = 0

    # --- helpers: generate shape + draw on mask + draw on image ----
    def generate_pts(stype):
        if stype == "triangle":      return random_triangle(size, size)
        if stype == "quadrilateral": return random_quadrilateral(size, size)
        return random_star(size, size)

    def draw_mask(tmp, stype, pts):
        if stype == "star":
            C = pts[0]
            for p in pts[1:]:
                cv2.line(tmp, tuple(C), tuple(p), 255, 3)
        else:
            cv2.fillPoly(tmp, [np.array(pts)], 255)
        return tmp

    def draw_shape(img, stype, pts, color):
        if stype == "triangle":      draw_triangle(img, pts, color)
        elif stype == "quadrilateral": draw_quadrilateral(img, pts, color)
        else:                        draw_star(img, pts, color)

    # --- main loop -------------------------------------------------
    while len(all_kps) < max_shapes * 6 and attempts < 60:
        attempts += 1

        stype = np.random.choice(shape_types)
        pts = generate_pts(stype)

        tmp = draw_mask(np.zeros_like(mask), stype, pts)

        # overlap check
        if np.any(mask & tmp):
            continue

        # draw accepted shape
        color = ensure_contrast(bg, random_color())
        draw_shape(img, stype, pts, color)

        # mask = mask OR tmp meaning the new shape becomes part of the overall mask
        mask |= tmp
        all_kps.extend(pts)

    return add_smoothed_noise(img), all_kps



# ===============================================================
# Visualization + saving
# ===============================================================


def save_keypoint_mask(keypoints, size, out_path, radius=3):
    """
    Create a black RGB image showing ONLY the keypoints as green dots.
    """
    mask = np.zeros((size, size, 3), dtype=np.uint8)  # RGB black background

    for x, y in keypoints:
        cv2.circle(mask, (int(x), int(y)), radius, (0, 255, 0), -1)  # green

    cv2.imwrite(out_path, mask)

def visualize_and_save(img, keypoints, out_path, radius: int = 4, color=(0, 0, 255)):
    """Save visualization with keypoints drawn."""
    vis = img.copy()
    for x, y in keypoints:
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    cv2.imwrite(out_path, vis)


def save_example(img, kps, folder, prefix, idx: int):
    base = f"{folder}/{prefix}_{idx:03d}"

    # original image
    cv2.imwrite(base + ".png", img)

    # keypoints as .npy
    np.save(base + ".npy", np.array(kps, dtype=np.int32))

    # 
    # visualize_and_save(img, kps, base + "_vis.png")

    # NEW: save keypoint mask (black background)
    save_keypoint_mask(kps, img.shape[0], base + "_mask.png")


# ===============================================================
# Dataset generation: original + homography
# ===============================================================

def generate_dataset(output_dir: str = "dataset", n: int = 30, size: int = 256):
    os.makedirs(output_dir, exist_ok=True)

    simple_shapes  = ["triangle", "quadrilateral", "star"]
    complex_shapes = ["chessboard", "cube"]
    all_groups     = simple_shapes + complex_shapes + ["multi"]

    # create subfolders: shape/original, shape/homography
    for shape in all_groups:
        os.makedirs(f"{output_dir}/{shape}/original", exist_ok=True)
        os.makedirs(f"{output_dir}/{shape}/homography", exist_ok=True)

    # -----------------------------------------------------------
    # 1) Simple shapes
    # -----------------------------------------------------------
    print("\nGenerating simple shapes (original + homography)...")
    for shape in simple_shapes:
        orig_folder = f"{output_dir}/{shape}/original"
        hom_folder  = f"{output_dir}/{shape}/homography"

        for i in range(n):
            # original
            img, kps = generate_simple_shape(shape, size)
            save_example(img, kps, orig_folder, shape, i)

            # homography-augmented (from same base)
            h_img, h_kps = augment_with_homography(img, kps, size=size)
            save_example(h_img, h_kps, hom_folder, shape, i)

    # -----------------------------------------------------------
    # 2) Complex shapes
    # -----------------------------------------------------------
    print("\nGenerating complex shapes (original + homography)...")
    for shape in complex_shapes:
        orig_folder = f"{output_dir}/{shape}/original"
        hom_folder  = f"{output_dir}/{shape}/homography"

        for i in range(n):
            img, kps = generate_simple_shape(shape, size)
            save_example(img, kps, orig_folder, shape, i)

            h_img, h_kps = augment_with_homography(img, kps, size=size)
            save_example(h_img, h_kps, hom_folder, shape, i)

    # -----------------------------------------------------------
    # 3) Multi-shape images
    # -----------------------------------------------------------
    print("\nGenerating multi-shape images (original + homography)...")
    orig_folder = f"{output_dir}/multi/original"
    hom_folder  = f"{output_dir}/multi/homography"

    for i in range(n):
        img, kps = generate_multiple_shapes(size=size, max_shapes=4)
        save_example(img, kps, orig_folder, "multi", i)

        h_img, h_kps = augment_with_homography(img, kps, size=size)
        save_example(h_img, h_kps, hom_folder, "multi", i)

    print("\nDataset generation complete!")
    print(f"Saved to: {output_dir}")


# ===============================================================
# RUN MAIN
# ===============================================================
if __name__ == "__main__":
    generate_dataset(output_dir="dataset", n=30, size=256)
