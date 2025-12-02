import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from model import build_superpoint_detector, extract_keypoints
from generate_dataset import SyntheticDataGenerator


def homographic_adaptation(model, image, num_homographies=99, threshold=0.015):
    """
    Apply homographic adaptation for robust keypoint detection.
    
    Args:
        model: Trained SuperPoint model
        image: Input image (H, W) or (H, W, 1)
        num_homographies: Number of random homographies to apply (default 99 + 1 original)
        threshold: Threshold for keypoint extraction
    
    Returns:
        keypoints: Detected keypoints (N, 2)
        scores: Confidence scores (N,)
    """
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    
    h, w = image.shape[:2]
    
    # Accumulator for probability maps
    prob_accumulator = np.zeros((h, w), dtype=np.float32)
    
    # Process original image
    img_norm = image.astype(np.float32) / 255.0
    output = model(img_norm[np.newaxis, ...], training=False)
    prob_map = output['prob_map'][0].numpy()
    prob_accumulator += prob_map
    
    # Process homography-transformed images
    for i in range(num_homographies):
        # Generate random homography
        margin = min(w, h) // 4
        src_points = np.float32([
            [np.random.randint(margin, w//2), np.random.randint(margin, h//2)],
            [np.random.randint(w//2, w-margin), np.random.randint(margin, h//2)],
            [np.random.randint(w//2, w-margin), np.random.randint(h//2, h-margin)],
            [np.random.randint(margin, w//2), np.random.randint(h//2, h-margin)]
        ])
        
        dst_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        rotation = np.random.randint(0, 4)
        dst_points = np.roll(dst_points, rotation, axis=0)
        
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Transform image
        img_transformed = cv2.warpPerspective(image, H, (w, h))
        img_transformed_norm = img_transformed.astype(np.float32) / 255.0
        
        # Get probability map
        output = model(img_transformed_norm[np.newaxis, ...], training=False)
        prob_map_transformed = output['prob_map'][0].numpy()
        
        # Inverse transform probability map
        H_inv = np.linalg.inv(H)
        prob_map_inv = cv2.warpPerspective(prob_map_transformed, H_inv, (w, h))
        
        prob_accumulator += prob_map_inv
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_homographies} homographies")
    
    # Average probability map
    prob_accumulator /= (num_homographies + 1)
    
    # Extract keypoints from averaged probability map
    prob_tensor = tf.convert_to_tensor(prob_accumulator)
    keypoints, scores = extract_keypoints(prob_tensor, threshold=threshold)
    
    return keypoints, scores, prob_accumulator


def evaluate_on_synthetic(model, generator, num_samples=10, threshold=0.015, 
                          output_dir='results/synthetic'):
    """Evaluate model on synthetic images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating on {num_samples} synthetic images...")
    
    for i in range(num_samples):
        # Generate sample
        img, gt_keypoints = generator.generate_sample()
        
        # Normalize and add batch dimension
        img_norm = img.astype(np.float32) / 255.0
        img_input = img_norm[..., np.newaxis][np.newaxis, ...]
        
        # Predict
        output = model(img_input, training=False)
        prob_map = output['prob_map'][0].numpy()
        
        # Extract keypoints
        prob_tensor = tf.convert_to_tensor(prob_map)
        pred_keypoints, scores = extract_keypoints(prob_tensor, threshold=threshold)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with ground truth
        img_gt = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for kp in gt_keypoints:
            cv2.circle(img_gt, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
        axes[0].imshow(img_gt)
        axes[0].set_title(f'Ground Truth ({len(gt_keypoints)} keypoints)')
        axes[0].axis('off')
        
        # Probability map
        axes[1].imshow(prob_map, cmap='hot')
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        
        # Detected keypoints
        img_pred = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for kp, score in zip(pred_keypoints, scores):
            cv2.circle(img_pred, tuple(kp.astype(int)), 3, (255, 0, 0), -1)
        axes[2].imshow(img_pred)
        axes[2].set_title(f'Detected ({len(pred_keypoints)} keypoints)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'synthetic_{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Results saved to {output_dir}/")


def evaluate_on_real_image(model, image_path, threshold=0.015, use_homography=False,
                           output_path='results/real_image_result.png'):
    """Evaluate model on real photograph."""
    print(f"\nEvaluating on real image: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Resize to match training dimensions (keep aspect ratio)
    h, w = img.shape
    target_h, target_w = 240, 320
    
    # Ensure dimensions are divisible by 8
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale) // 8 * 8
    new_h = int(h * scale) // 8 * 8
    img_resized = cv2.resize(img, (new_w, new_h))
    
    if use_homography:
        print("Using homographic adaptation (this may take a few minutes)...")
        keypoints, scores, prob_map = homographic_adaptation(
            model, img_resized, num_homographies=99, threshold=threshold
        )
        title_suffix = " (with Homographic Adaptation)"
    else:
        # Normalize and predict
        img_norm = img_resized.astype(np.float32) / 255.0
        img_input = img_norm[..., np.newaxis][np.newaxis, ...]
        
        output = model(img_input, training=False)
        prob_map = output['prob_map'][0].numpy()
        
        # Extract keypoints
        prob_tensor = tf.convert_to_tensor(prob_map)
        keypoints, scores = extract_keypoints(prob_tensor, threshold=threshold)
        title_suffix = ""
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img_resized, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Probability map
    im = axes[1].imshow(prob_map, cmap='hot')
    axes[1].set_title('Probability Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Detected keypoints
    img_result = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    for kp, score in zip(keypoints, scores):
        # Color based on confidence
        color = (0, int(255 * score), int(255 * (1 - score)))
        cv2.circle(img_result, tuple(kp.astype(int)), 3, color, -1)
        cv2.circle(img_result, tuple(kp.astype(int)), 5, (0, 255, 0), 1)
    
    axes[2].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Detected Keypoints ({len(keypoints)}){title_suffix}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Result saved to {output_path}")
    plt.close()
    
    return keypoints, scores


def compare_with_without_homography(model, image_path, threshold=0.015,
                                   output_path='results/homography_comparison.png'):
    """Compare detection with and without homographic adaptation."""
    print(f"\nComparing with/without homographic adaptation on: {image_path}")
    
    # Load and preprocess image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    h, w = img.shape
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Without homography
    print("Detection without homographic adaptation...")
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = img_norm[..., np.newaxis][np.newaxis, ...]
    output = model(img_input, training=False)
    prob_map_no_homo = output['prob_map'][0].numpy()
    
    prob_tensor = tf.convert_to_tensor(prob_map_no_homo)
    kpts_no_homo, scores_no_homo = extract_keypoints(prob_tensor, threshold=threshold)
    
    # With homography
    print("Detection with homographic adaptation...")
    kpts_homo, scores_homo, prob_map_homo = homographic_adaptation(
        model, img_resized, num_homographies=99, threshold=threshold
    )
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Without homography
    axes[0, 0].imshow(img_resized, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(prob_map_no_homo, cmap='hot')
    axes[0, 1].set_title('Probability Map (No Homography)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    img_no_homo = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    for kp in kpts_no_homo:
        cv2.circle(img_no_homo, tuple(kp.astype(int)), 3, (255, 0, 0), -1)
        cv2.circle(img_no_homo, tuple(kp.astype(int)), 5, (0, 255, 0), 1)
    axes[0, 2].imshow(cv2.cvtColor(img_no_homo, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Detected ({len(kpts_no_homo)} keypoints)')
    axes[0, 2].axis('off')
    
    # Row 2: With homography
    axes[1, 0].imshow(img_resized, cmap='gray')
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    im2 = axes[1, 1].imshow(prob_map_homo, cmap='hot')
    axes[1, 1].set_title('Probability Map (With Homography)')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    img_homo = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
    for kp in kpts_homo:
        cv2.circle(img_homo, tuple(kp.astype(int)), 3, (255, 0, 0), -1)
        cv2.circle(img_homo, tuple(kp.astype(int)), 5, (0, 255, 0), 1)
    axes[1, 2].imshow(cv2.cvtColor(img_homo, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Detected ({len(kpts_homo)} keypoints)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("SuperPoint Keypoint Detector Evaluation")
    print("=" * 60)
    
    # Load trained model
    model_path = 'checkpoints/best_model.h5'
    print(f"\nLoading model from {model_path}...")
    
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={'detector_loss': None},
            compile=False
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model has been trained and saved.")
        exit(1)
    
    # Configuration
    THRESHOLD = 0.015  # Adjust based on desired sensitivity
    
    # Evaluate on synthetic images
    generator = SyntheticDataGenerator(img_height=240, img_width=320)
    evaluate_on_synthetic(model, generator, num_samples=10, threshold=THRESHOLD)
    
    # Evaluate on real image (if available)
    real_image_path = 'test_image.jpg'  # Replace with your test image
    if Path(real_image_path).exists():
        # Without homographic adaptation
        evaluate_on_real_image(
            model, real_image_path,
            threshold=THRESHOLD,
            use_homography=False,
            output_path='results/real_image_no_homography.png'
        )
        
        # With homographic adaptation
        evaluate_on_real_image(
            model, real_image_path,
            threshold=THRESHOLD,
            use_homography=True,
            output_path='results/real_image_with_homography.png'
        )
        
        # Comparison
        compare_with_without_homography(
            model, real_image_path,
            threshold=THRESHOLD,
            output_path='results/homography_comparison.png'
        )
    else:
        print(f"\nTest image not found at {real_image_path}")
        print("Please provide a test image to evaluate on real photographs.")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    print("\nResults saved to results/ directory")