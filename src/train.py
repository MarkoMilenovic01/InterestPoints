import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from model import build_superpoint_detector, keypoints_to_grid, detector_loss
from generate_dataset import SyntheticDataGenerator


class KeypointDataGenerator(keras.utils.Sequence):
    """Data generator for training."""
    
    def __init__(self, generator, batch_size=32, img_shape=(240, 320), 
                 samples_per_epoch=1000, use_saved_data=False, data_dir=None):
        self.generator = generator
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.samples_per_epoch = samples_per_epoch
        self.use_saved_data = use_saved_data
        self.data_dir = Path(data_dir) if data_dir else None
        
        if use_saved_data and self.data_dir:
            self.image_files = sorted(list(self.data_dir.glob("img_*.png")))
            self.keypoint_files = sorted(list(self.data_dir.glob("kpts_*.npy")))
            self.samples_per_epoch = len(self.image_files)
    
    def __len__(self):
        return self.samples_per_epoch // self.batch_size
    
    def __getitem__(self, idx):
        batch_images = []
        batch_grids = []
        
        for _ in range(self.batch_size):
            if self.use_saved_data:
                # Load from disk
                img_idx = np.random.randint(0, len(self.image_files))
                img = cv2.imread(str(self.image_files[img_idx]), cv2.IMREAD_GRAYSCALE)
                keypoints = np.load(str(self.keypoint_files[img_idx]))
            else:
                # Generate on-the-fly
                img, keypoints = self.generator.generate_sample()
            
            # Normalize image
            img = img.astype(np.float32) / 255.0
            
            # Convert keypoints to grid
            grid = keypoints_to_grid(keypoints, self.img_shape)
            
            batch_images.append(img)
            batch_grids.append(grid)
        
        batch_images = np.array(batch_images)[..., np.newaxis]  # Add channel dimension
        batch_grids = np.array(batch_grids)
        
        return batch_images, batch_grids


def train_model(model, train_generator, epochs=200, initial_epoch=0, 
                log_dir='logs', checkpoint_dir='checkpoints'):
    """Train the SuperPoint detector model."""
    
    # Create directories
    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        optimizer=optimizer,
        loss={'logits': detector_loss},
        metrics={'logits': ['accuracy']}
    )
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(log_dir / timestamp),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'model_epoch_{epoch:03d}.h5'),
        save_weights_only=False,
        save_freq='epoch',
        period=10  # Save every 10 epochs
    )
    
    best_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / 'best_model.h5'),
        save_weights_only=False,
        save_best_only=True,
        monitor='loss',
        mode='min'
    )
    
    # Learning rate scheduler
    def lr_schedule(epoch, lr):
        """Decay learning rate at specific epochs."""
        if epoch in [100, 150]:
            return lr * 0.1
        return lr
    
    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # Custom callback to save training history
    class HistoryCallback(keras.callbacks.Callback):
        def __init__(self, filepath):
            super().__init__()
            self.filepath = filepath
            self.history_data = {'loss': [], 'accuracy': [], 'lr': []}
        
        def on_epoch_end(self, epoch, logs=None):
            self.history_data['loss'].append(logs.get('loss'))
            self.history_data['accuracy'].append(logs.get('logits_accuracy'))
            self.history_data['lr'].append(float(keras.backend.get_value(self.model.optimizer.lr)))
            
            # Save history
            np.save(self.filepath, self.history_data)
    
    history_callback = HistoryCallback(str(checkpoint_dir / 'training_history.npy'))
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[
            tensorboard_callback,
            checkpoint_callback,
            best_checkpoint_callback,
            lr_callback,
            history_callback
        ],
        verbose=1
    )
    
    return history


def plot_training_history(history_path='checkpoints/training_history.npy', 
                          output_path='training_curves.png'):
    """Plot training loss and accuracy curves."""
    history_data = np.load(history_path, allow_pickle=True).item()
    
    epochs = range(1, len(history_data['loss']) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curve
    ax1.plot(epochs, history_data['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, history_data['accuracy'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Learning rate curve
    ax3.plot(epochs, history_data['lr'], 'r-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()


def visualize_samples(generator, num_samples=5, output_path='training_samples.png'):
    """Visualize training samples with keypoints."""
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        img, keypoints = generator.generate_sample()
        
        # Draw keypoints
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for kp in keypoints:
            cv2.circle(img_color, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
            cv2.circle(img_color, tuple(kp.astype(int)), 5, (0, 0, 255), 1)
        
        ax.imshow(img_color)
        ax.set_title(f'Sample {i+1}\n{len(keypoints)} keypoints')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training samples saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    IMG_HEIGHT = 240
    IMG_WIDTH = 320
    BATCH_SIZE = 16  # Adjust based on GPU memory
    EPOCHS = 200  # Can reduce for faster training
    SAMPLES_PER_EPOCH = 1000
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("=" * 60)
    print("SuperPoint Keypoint Detector Training")
    print("=" * 60)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\nNo GPU found. Training will use CPU (slower).")
    
    # Create synthetic data generator
    print("\nInitializing data generator...")
    generator = SyntheticDataGenerator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    
    # Visualize training samples
    print("Generating sample visualizations...")
    visualize_samples(generator, num_samples=5)
    
    # Create data generator for training
    train_generator = KeypointDataGenerator(
        generator=generator,
        batch_size=BATCH_SIZE,
        img_shape=(IMG_HEIGHT, IMG_WIDTH),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        use_saved_data=False  # Set to True if you want to use pre-generated dataset
    )
    
    print(f"\nTraining configuration:")
    print(f"  Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Samples per epoch: {SAMPLES_PER_EPOCH}")
    print(f"  Steps per epoch: {len(train_generator)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_superpoint_detector(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Print model summary
    print("\nModel architecture:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    history = train_model(
        model=model,
        train_generator=train_generator,
        epochs=EPOCHS,
        log_dir='logs',
        checkpoint_dir='checkpoints'
    )
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_history()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nModel saved to: checkpoints/best_model.h5")
    print("Training logs saved to: logs/")
    print("View training progress with: tensorboard --logdir logs/")