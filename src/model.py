import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ResNetBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        self.shortcut_conv = None
        self.shortcut_bn = None
        
        self.relu_out = layers.ReLU()
    
    def build(self, input_shape):
        # If input channels != output channels, add projection shortcut
        if input_shape[-1] != self.filters:
            self.shortcut_conv = layers.Conv2D(self.filters, 1, padding='same')
            self.shortcut_bn = layers.BatchNormalization()
        super(ResNetBlock, self).build(input_shape)
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
        
        x = x + shortcut
        x = self.relu_out(x)
        
        return x


def build_encoder(input_shape=(240, 320, 1)):
    """Build encoder network following assignment specification."""
    inputs = layers.Input(shape=input_shape)
    
    # First stage: 2 ResNet blocks, 64 channels
    x = ResNetBlock(64)(inputs)
    x = ResNetBlock(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Second stage: 2 ResNet blocks, 64 channels
    x = ResNetBlock(64)(x)
    x = ResNetBlock(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Third stage: 2 ResNet blocks, 128 channels
    x = ResNetBlock(128)(x)
    x = ResNetBlock(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Fourth stage: 2 ResNet blocks, 128 channels
    x = ResNetBlock(128)(x)
    x = ResNetBlock(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return keras.Model(inputs=inputs, outputs=x, name='encoder')


def build_detector_head(encoder_output_shape):
    """Build detector head following assignment specification."""
    inputs = layers.Input(shape=encoder_output_shape)
    
    # Detector head
    x = layers.Conv2D(256, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Output 65 channels (64 positions + 1 no-keypoint class)
    x = layers.Conv2D(65, 1, padding='same')(x)
    
    return keras.Model(inputs=inputs, outputs=x, name='detector_head')


def logits_to_prob_map(logits):
    """Convert logits to probability map.
    
    Args:
        logits: Tensor of shape (batch, H/8, W/8, 65)
    
    Returns:
        prob_map: Tensor of shape (batch, H, W)
    """
    # Apply softmax
    prob = tf.nn.softmax(logits, axis=-1)
    
    # Remove dustbin channel (last channel)
    prob = prob[..., :-1]  # Shape: (batch, H/8, W/8, 64)
    
    # Reshape to full resolution
    # Method 1: Using tf.nn.depth_to_space
    prob_map = tf.nn.depth_to_space(prob, block_size=8)
    
    return prob_map


def build_superpoint_detector(input_shape=(240, 320, 1)):
    """Build complete SuperPoint detector model."""
    # Build encoder
    encoder = build_encoder(input_shape)
    
    # Build detector head
    encoder_output_shape = encoder.output_shape[1:]
    detector_head = build_detector_head(encoder_output_shape)
    
    # Complete model
    inputs = layers.Input(shape=input_shape)
    features = encoder(inputs)
    logits = detector_head(features)
    
    # For training, we output logits
    # For inference, we convert to probability map
    prob_map = layers.Lambda(lambda x: logits_to_prob_map(x), name='prob_map')(logits)
    
    model = keras.Model(inputs=inputs, outputs={'logits': logits, 'prob_map': prob_map}, 
                       name='superpoint_detector')
    
    return model


def keypoints_to_grid(keypoints, img_shape):
    """Convert keypoints to grid representation for training.
    
    Args:
        keypoints: Array of shape (N, 2) with (x, y) coordinates
        img_shape: Tuple (H, W)
    
    Returns:
        grid: Array of shape (H/8, W/8, 65) for cross-entropy loss
    """
    H, W = img_shape
    H_cell, W_cell = H // 8, W // 8
    
    # Initialize grid with dustbin class (channel 64)
    grid = np.zeros((H_cell, W_cell, 65), dtype=np.float32)
    grid[..., 64] = 1.0  # All cells start as "no keypoint"
    
    for kp in keypoints:
        x, y = kp
        # Determine which cell this keypoint belongs to
        cell_x = int(x // 8)
        cell_y = int(y // 8)
        
        # Position within cell
        local_x = int(x % 8)
        local_y = int(y % 8)
        
        # Channel index (row-major order within 8x8 cell)
        channel = local_y * 8 + local_x
        
        if 0 <= cell_y < H_cell and 0 <= cell_x < W_cell and 0 <= channel < 64:
            # If dustbin was set, remove it
            if grid[cell_y, cell_x, 64] == 1.0:
                grid[cell_y, cell_x, 64] = 0.0
            
            # Set the corresponding position
            grid[cell_y, cell_x, channel] = 1.0
    
    # Handle multiple keypoints in same cell (keep one randomly)
    for i in range(H_cell):
        for j in range(W_cell):
            active_channels = np.where(grid[i, j, :64] > 0)[0]
            if len(active_channels) > 1:
                # Keep one random channel
                keep_channel = np.random.choice(active_channels)
                grid[i, j, :64] = 0.0
                grid[i, j, keep_channel] = 1.0
            
            # Ensure either one keypoint or dustbin is active
            if np.sum(grid[i, j, :]) == 0:
                grid[i, j, 64] = 1.0
    
    return grid


def detector_loss(y_true, y_pred):
    """
    Categorical cross-entropy loss for keypoint detection.
    
    Args:
        y_true: Ground truth grid (H/8, W/8, 65) with one-hot encoding
        y_pred: Predicted logits (H/8, W/8, 65)
    
    Returns:
        loss: Scalar loss value
    """
    # Categorical cross-entropy with logits
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    
    # Average over all cells
    return tf.reduce_mean(loss)


def extract_keypoints(prob_map, threshold=0.015, nms_size=4):
    """Extract keypoints from probability map.
    
    Args:
        prob_map: Probability map of shape (H, W)
        threshold: Minimum probability threshold
        nms_size: Size for non-maximum suppression
    
    Returns:
        keypoints: Array of shape (N, 2) with (x, y) coordinates
        scores: Array of shape (N,) with confidence scores
    """
    # Apply threshold
    mask = prob_map > threshold
    
    # Non-maximum suppression using max pooling
    prob_map_max = tf.nn.max_pool2d(
        prob_map[None, ..., None],
        ksize=nms_size,
        strides=1,
        padding='SAME'
    )[0, ..., 0]
    
    # Keep only local maxima
    mask = mask & (prob_map == prob_map_max)
    
    # Extract coordinates
    keypoints = tf.where(mask)
    scores = tf.gather_nd(prob_map, keypoints)
    
    # Convert to (x, y) format
    keypoints = tf.cast(keypoints[:, ::-1], tf.float32)  # Swap to (x, y)
    
    return keypoints.numpy(), scores.numpy()


if __name__ == "__main__":
    # Test model building
    model = build_superpoint_detector(input_shape=(240, 320, 1))
    model.summary()
    
    print("\nModel output shapes:")
    dummy_input = np.random.randn(1, 240, 320, 1).astype(np.float32)
    outputs = model(dummy_input)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Probability map shape: {outputs['prob_map'].shape}")