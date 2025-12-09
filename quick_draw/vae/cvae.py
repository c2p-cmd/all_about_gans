"""
Conditional Variational Autoencoder (CVAE) for Google Quick Draw Dataset
Training script with TensorBoard support and periodic checkpoint saving
"""

import argparse
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from dataset import categories


# ============================================================================
# CVAE Model Architecture
# ============================================================================


class Encoder(nn.Module):
    def __init__(self, latent_dim=16, max_filters=64, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

        # Better architecture: 28x28 -> 14x14 -> 7x7 -> 7x7
        self.conv1 = nn.Conv2d(1, max_filters // 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            max_filters // 4, max_filters // 2, kernel_size=3, stride=2, padding=1
        )
        # Changed: stride=1 to keep 7x7
        self.conv3 = nn.Conv2d(
            max_filters // 2, max_filters, kernel_size=3, stride=1, padding=1
        )

        self.cond_conv = nn.Conv2d(max_filters, max_filters, kernel_size=1)

        self.flatten_dim = 7 * 7 * max_filters  # Now correct
        self.fc_mu = nn.Linear(self.flatten_dim + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + num_classes, latent_dim)

    def __call__(self, x, labels):
        # x: (batch, 28, 28, 1) already in NHWC
        h = nn.relu(self.conv1(x))  # -> 14x14
        h = nn.relu(self.conv2(h))  # -> 7x7
        h = nn.relu(self.conv3(h))  # -> 7x7 (stride=1)
        h = nn.relu(self.cond_conv(h))  # -> 7x7

        h = mx.reshape(h, (h.shape[0], -1))

        labels_one_hot = mx.zeros((labels.shape[0], self.num_classes))
        labels_one_hot[mx.arange(labels.shape[0]), labels] = 1
        h = mx.concatenate([h, labels_one_hot], axis=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, max_filters=64, num_classes=8):
        super().__init__()
        self.num_classes = num_classes
        self.max_filters = max_filters

        self.fc = nn.Linear(
            latent_dim + num_classes,
            7 * 7 * max_filters,
        )

        self.conv1 = nn.Conv2d(
            max_filters,
            max_filters // 2,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            max_filters // 2,
            max_filters // 4,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            max_filters // 4,
            1,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, z, labels):
        labels_one_hot = mx.zeros((labels.shape[0], self.num_classes))
        labels_one_hot[mx.arange(labels.shape[0]), labels] = 1
        z = mx.concatenate([z, labels_one_hot], axis=1)

        h = self.fc(z)
        h = mx.reshape(h, (z.shape[0], 7, 7, self.max_filters))

        # 7x7 -> 14x14 -> 28x28
        h = nn.relu(self.conv1(self._upsample(h, 2)))
        h = nn.relu(self.conv2(self._upsample(h, 2)))
        h = mx.sigmoid(self.conv3(h))

        return h

    def _upsample(self, x, scale_factor):
        """Nearest neighbor upsampling"""
        B, H, W, C = x.shape
        x = mx.repeat(x, scale_factor, axis=1)
        x = mx.repeat(x, scale_factor, axis=2)
        return x


class CVAE(nn.Module):
    """
    Complete Conditional Variational Autoencoder
    """

    def __init__(self, latent_dim=16, max_filters=64, num_classes=8):
        super().__init__()
        self.encoder = Encoder(latent_dim, max_filters, num_classes)
        self.decoder = Decoder(latent_dim, max_filters, num_classes)
        self.latent_dim = latent_dim

    def __call__(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, labels)
        return x_recon, mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = mx.exp(0.5 * logvar)
        epsilon = mx.random.normal(mu.shape)
        return mu + std * epsilon

    def sample(self, num_samples, labels):
        """Generate samples from the model"""
        z = mx.random.normal((num_samples, self.latent_dim))
        return self.decoder(z, labels)


# ============================================================================
# Loss Function
# ============================================================================


def vae_loss(x_recon, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence
    """
    # Reconstruction loss (binary cross-entropy)
    x_flat = mx.reshape(x, (x.shape[0], -1))
    x_recon_flat = mx.reshape(x_recon, (x_recon.shape[0], -1))

    recon_loss = -mx.sum(
        x_flat * mx.log(x_recon_flat + 1e-8)
        + (1 - x_flat) * mx.log(1 - x_recon_flat + 1e-8),
        axis=1,
    )

    # KL divergence
    kl_div = -0.5 * mx.sum(1 + logvar - mu**2 - mx.exp(logvar), axis=1)

    beta = 1.0
    kl_div = beta * kl_div

    return mx.mean(recon_loss + kl_div)


# ============================================================================
# Data Loading for Google Quick Draw
# ============================================================================


def load_quickdraw_data(data_dir, categories, samples_per_class=10000):
    """
    Load Quick Draw data from numpy files
    Returns normalized images and labels
    """
    all_images = []
    all_labels = []

    for idx, category in enumerate(categories):
        filepath = os.path.join(data_dir, f"{category}.npy")

        # Load and preprocess
        data = np.load(filepath)[:samples_per_class]
        data = data.astype(np.float32) / 255.0  # Normalize to [0, 1]
        data = data.reshape(-1, 28, 28, 1)  # Reshape to (N, H, W, C)

        all_images.append(data)
        all_labels.append(np.full(len(data), idx, dtype=np.int32))

        print(f"Loaded {len(data)} samples of {category}")

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    return mx.array(images), mx.array(labels)


def batch_iterate(images, labels, batch_size, shuffle=True):
    """
    Create batches from the dataset
    """
    num_samples = images.shape[0]
    indices = mx.arange(num_samples)

    if shuffle:
        indices = mx.random.permutation(indices)

    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : min(i + batch_size, num_samples)]
        yield images[batch_indices], labels[batch_indices]


# ============================================================================
# Training and Evaluation
# ============================================================================


def train_epoch(model, optimizer, images, labels, batch_size):
    """Train for one epoch"""
    total_loss = 0.0
    num_batches = 0

    def loss_fn(model, x, y):
        x_recon, mu, logvar = model(x, y)
        return vae_loss(x_recon, x, mu, logvar)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for batch_x, batch_y in batch_iterate(images, labels, batch_size):
        loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_samples(model, categories, epoch, save_dir, num_samples=6):
    """Generate and save sample images"""
    samples_dir = Path(save_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Generate one sample per class
    labels = mx.array(list(range(len(categories))))
    samples = model.sample(len(categories), labels)

    # Create image grid
    samples_np = np.array(samples.squeeze())
    samples_np = (samples_np * 255).astype(np.uint8)

    # Save individual images
    for idx, (sample, category) in enumerate(zip(samples_np, categories)):
        img = Image.fromarray(sample, mode="L")
        img.save(samples_dir / f"epoch_{epoch:03d}_{category}.png")

    # Create grid
    grid_img = np.hstack(samples_np)
    grid = Image.fromarray(grid_img, mode="L")
    grid.save(samples_dir / f"epoch_{epoch:03d}_grid.png")

    return grid_img


def save_reconstructions(
    model,
    images,
    labels,
    categories,
    epoch,
    save_dir,
    num_samples=len(categories),
):
    """Save reconstruction examples"""
    recon_dir = Path(save_dir) / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    # Convert to NumPy for boolean indexing
    images_np = np.array(images)
    labels_np = np.array(labels)

    samples_per_class = num_samples // len(categories)
    selected_imgs = []
    selected_labels = []

    for class_idx in range(len(categories)):
        # Boolean indexing works in NumPy
        class_mask = labels_np == class_idx
        class_imgs = images_np[class_mask][:samples_per_class]

        selected_imgs.append(class_imgs)
        selected_labels.extend([class_idx] * len(class_imgs))

    # Convert back to MLX arrays
    test_imgs = mx.array(np.concatenate(selected_imgs, axis=0))
    test_labels = mx.array(selected_labels)

    recon, _, _ = model(test_imgs, test_labels)

    test_np = np.array(test_imgs.squeeze())
    recon_np = np.array(recon.squeeze())

    plt.figure(figsize=(num_samples, 4))
    for i in range(num_samples):
        # Original
        ax = plt.subplot(2, num_samples, i + 1)
        plt.imshow(test_np[i], cmap="gray")
        plt.title(categories[test_labels[i].item()])
        plt.axis("off")

        # Reconstruction
        ax = plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(recon_np[i], cmap="gray")
        plt.title(categories[test_labels[i].item()])
        plt.axis("off")
    plt.savefig(recon_dir / f"epoch_{epoch:03d}_reconstructions.png")
    plt.close()


# ============================================================================
# Main Training Script
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train CVAE on Google Quick Draw")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./quick_draw_data",
        help="Directory for reading dataset",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models_weights",
        help="Directory to save models and samples",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default=None,
        help="Directory to load model weights from",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./runs", help="TensorBoard log directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=16, help="Dimensionality of latent space"
    )
    parser.add_argument(
        "--max_filters",
        type=int,
        default=64,
        help="Maximum number of convolutional filters",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=10000,
        help="Number of samples to use per class",
    )
    parser.add_argument("--seed", type=int, default=44, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # Animal categories for Quick Draw

    print("=" * 70)
    print("CVAE Training on Google Quick Draw")
    print("=" * 70)
    print(f"Categories: {categories}")
    print(f"Device: {mx.default_device()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dimensions: {args.latent_dim}")
    print(f"Max filters: {args.max_filters}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)

    # Create directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Download and load data
    # download_quickdraw_data(args.data_dir, categories)
    train_images, train_labels = load_quickdraw_data(
        args.data_dir,
        categories,
        args.samples_per_class,
    )

    print(f"\nDataset loaded: {train_images.shape[0]} total samples")
    print(f"Image shape: {train_images.shape[1:]}\n")

    # Initialize model
    model = CVAE(
        latent_dim=args.latent_dim,
        max_filters=args.max_filters,
        num_classes=len(categories),
    )
    if args.load_dir is not None:
        weights_path = Path(args.load_dir)
        model.load_weights(str(weights_path))
        print(f"Loaded model weights from {weights_path}\n")

    # Count parameters
    from mlx.utils import tree_flatten

    num_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"Number of trainable parameters: {num_params / 1e6:.4f}M\n")

    # Optimizer
    optimizer = optim.Adam(learning_rate=args.lr)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        avg_loss = train_epoch(
            model,
            optimizer,
            train_images,
            train_labels,
            args.batch_size,
        )

        epoch_time = time.time() - start_time
        throughput = len(train_images) / epoch_time

        # Log to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Throughput", throughput, epoch)

        print(
            f"Epoch {epoch:3d} | Loss {avg_loss:8.2f} | "
            f"Throughput {throughput:7.2f} samples/s | Time {epoch_time:5.1f}s"
        )

        # Save checkpoints and samples every 10 epochs
        if epoch % 10 == 0:
            # Save model weights
            weights_path = Path(args.save_dir) / f"cvae_epoch_{epoch:03d}.safetensors"
            model.save_weights(str(weights_path))
            print(f"  → Saved weights to {weights_path}")

            # Generate and save samples
            grid_img = save_samples(model, categories, epoch, args.save_dir)
            save_reconstructions(
                model,
                train_images,
                train_labels,
                categories,
                epoch,
                args.save_dir,
            )

            # Log images to TensorBoard
            writer.add_image("Generated Samples", grid_img, epoch, dataformats="HW")

            print(f"  → Saved samples and reconstructions")

    # Save final model
    final_path = Path(args.save_dir) / "cvae_final.safetensors"
    model.save_weights(str(final_path))
    print(f"\nTraining complete! Final model saved to {final_path}")

    writer.close()


if __name__ == "__main__":
    import time

    start_time = time.time_ns()
    main()
    end_time = time.time_ns()

    elapsed_s = (end_time - start_time) / 1_00_00_00_000

    print("Time taken: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_s)))
