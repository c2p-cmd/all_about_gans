"""
C-DCGAN implementation using MLX to generate Fashion MNIST-like images.

This script defines and trains a Conditional - Deep Convolutional GAN (C-DCGAN) on the Fashion MNIST dataset.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from keras.datasets import fashion_mnist
from tqdm.auto import tqdm
import time

# --- 1. Load and Prepare Data ---

# Load Fashion MNIST
(X_train, y_train), (_, _) = fashion_mnist.load_data()
# Normalize to [-1, 1] to match the generator's tanh output
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# Add a channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
X_train = np.expand_dims(
    X_train,
    axis=-1,
)
# Convert to MLX array
X_train = mx.array(X_train)
y_train = mx.array(y_train)

print(f"Training data shape: {X_train.shape} | Labels shape: {y_train.shape}")

# --- 2. Define Hyperparameters ---

latent_dim = 100  # Size of the input noise vector
num_classes = 10  # Number of classes in MNIST
embedding_dim = 16  # Dimension of label embeddings
batch_size = 128
lr_g = 2.1e-4  # Learning rate for Generator
lr_d = 1.5e-4  # Learning rate for Discriminator
beta1 = 0.5  # Adam optimizer beta1
epochs = 60  # Number of training epochs

# --- 3. Define Models ---
mx.random.seed(44)


class Generator(nn.Module):
    """
    Takes a latent vector (z) and class label (c) and upsamples it to a 28x28x1 image.

    :param latent_dim: Dimension of the input noise vector.
    :param num_classes: Number of classes for conditional generation.
    :param embedding_dim: Dimension of the label embedding.
    """

    def __init__(self, latent_dim, num_classes, embedding_dim):
        super().__init__()

        # Embedding layer for class labels
        self.label_embedder = nn.Embedding(num_classes, embedding_dim)

        # Project latent vector to an initial 7x7x256 feature map
        self.fc = nn.Linear(latent_dim + embedding_dim, 7 * 7 * 256)

        # Upsampling block 1: 7x7 -> 7x7 (stride 1)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm(128)

        # Upsampling block 2: 7x7 -> 14x14 (stride 2)
        self.conv2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.bn2 = nn.BatchNorm(64)

        # Upsampling block 3: 14x14 -> 28x28 (stride 2)
        self.conv3 = nn.ConvTranspose2d(
            64,
            1,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

    def __call__(self, z, c):
        inputs = mx.concat([self.label_embedder(c), z], axis=1)

        x = self.fc(inputs)
        x = mx.reshape(x, (x.shape[0], 7, 7, 256))  # Reshape to (N, H, W, C)
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))

        # Final layer uses tanh to output values between [-1, 1]
        x = mx.tanh(self.conv3(x))
        return x


class Discriminator(nn.Module):
    """
    Takes a 28x28x1 image AND a class label (c) and classifies it as real (1) or fake (0).

    :param num_classes: Number of classes for conditional generation.
    :param embedding_dim: Dimension of the label embedding.
    """

    def __init__(self, num_classes, embedding_dim):
        super().__init__()

        # Embedding layer for class labels
        self.label_embedder = nn.Embedding(num_classes, embedding_dim)

        # Downsampling block 1: 28x28 -> 14x14
        self.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        # Downsampling block 2: 14x14 -> 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.ln2 = nn.LayerNorm(128)

        # Classifier head: 7*7*128 -> 1 (a single logit)
        self.fc = nn.Linear(7 * 7 * 128 + embedding_dim, 1)

    def __call__(self, x, c):
        # Pass (N, 28, 28, 1) image through CNN
        x = nn.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = nn.leaky_relu(self.ln2(self.conv2(x)), negative_slope=0.2)
        # Flatten image features to (N, 7*7*128)
        x_features = mx.reshape(x, (x.shape[0], -1))

        # Get label embedding (N, embedding_dim)
        c_embed = self.label_embedder(c)

        # Combine image features and label embedding
        inputs = mx.concatenate([x_features, c_embed], axis=-1)

        # Classify the combined vector
        x = self.fc(inputs)
        return x


# --- 4. Define Loss Functions ---

# What to do for Conv weights: N(0.0, 0.02)
init_conv_weight = nn.init.normal(mean=0.0, std=0.02)

# What to do for BatchNorm weights: N(1.0, 0.02)
init_bn_weight = nn.init.normal(mean=1.0, std=0.02)


# Filter for Conv/ConvT weights
def is_conv_weight(module: nn.Module, key: str, val) -> bool:
    return isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and key == "weight"


# Filter for BatchNorm weights
def is_bn_weight(module: nn.Module, key: str, val) -> bool:
    return isinstance(module, nn.BatchNorm) and key == "weight"


def discriminator_loss(real_output, fake_output):
    """Calculates D's loss, pushing real scores to 1 and fake scores to 0."""
    real_loss = nn.losses.binary_cross_entropy(
        real_output,
        mx.ones_like(real_output),  # Label smoothing for real images
        with_logits=True,
    )

    fake_loss = nn.losses.binary_cross_entropy(
        fake_output,
        mx.zeros_like(fake_output),
        with_logits=True,
    )
    return (real_loss + fake_loss) * 0.5


def generator_loss(fake_output):
    """Calculates G's loss, pushing fake scores to 1 (to fool D)."""
    return nn.losses.binary_cross_entropy(
        fake_output,
        mx.ones_like(fake_output),
        with_logits=True,
    )


# --- 5. Setup Models, Optimizers, and Grad Functions ---

generator = Generator(latent_dim, num_classes, embedding_dim)
generator.apply(init_conv_weight, is_conv_weight)
generator.apply(init_bn_weight, is_bn_weight)

discriminator = Discriminator(num_classes, embedding_dim)
discriminator.apply(init_conv_weight, is_conv_weight)

# Initialize parameters
mx.eval(generator.parameters(), discriminator.parameters())

opt_g = optim.Adam(learning_rate=lr_g, betas=[beta1, 0.999])
opt_d = optim.Adam(learning_rate=lr_d, betas=[beta1, 0.999])


# Loss function for the discriminator's gradient calculation
def d_loss_fn(real_images, real_labels, z, fake_labels):
    # Generate fake images
    fake_images = generator(z, fake_labels)

    # Detach fake_images to stop gradients flowing back to G
    fake_images = mx.stop_gradient(fake_images)

    # Get D's scores for real and fake
    real_output = discriminator(real_images, real_labels)
    fake_output = discriminator(fake_images, fake_labels)

    return discriminator_loss(real_output, fake_output)


# Loss function for the generator's gradient calculation
def g_loss_fn(z, labels):
    # Generate fake images (gradients WILL flow through G)
    fake_images = generator(z, labels)

    # Get D's score for fake (gradients WILL flow through D, but D's weights are frozen)
    fake_output = discriminator(fake_images, labels)

    # Calculate G's loss (how well it fooled D)
    return generator_loss(fake_output)


# Create the functions that compute loss and gradients
d_val_and_grad_fn = nn.value_and_grad(discriminator, d_loss_fn)
g_val_and_grad_fn = nn.value_and_grad(generator, g_loss_fn)

# --- 6. Training Loop ---

num_batches = len(X_train) // batch_size
losses = {"epoch": [], "batch": [], "d_loss": [], "g_loss": []}

start_time_training = time.time()

print("Starting Training...")
for epoch in range(epochs):
    start_time_epoch = time.time()
    for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        # Sample a batch of real images
        idx = mx.random.randint(0, len(X_train), (batch_size,))
        real_images = X_train[idx]
        real_labels = y_train[idx]

        # ===================================================================
        # START: (1) Train Discriminator
        # ===================================================================

        # Set modes: D trains (updates BN stats), G is for inference
        discriminator.train()
        generator.eval()

        # Sample noise for D step
        z_d = mx.random.normal((batch_size, latent_dim))
        fake_labels_d = mx.random.randint(0, num_classes, (batch_size,))

        # Calculate loss and gradients for D
        d_loss, d_grads = d_val_and_grad_fn(
            real_images,
            real_labels,
            z_d,
            fake_labels_d,
        )

        # Update D's weights
        opt_d.update(discriminator, d_grads)

        # Eval to ensure ops complete before next step
        mx.eval(discriminator.parameters(), generator.parameters())

        # ===================================================================
        # START: (2) Train Generator
        # ===================================================================

        # Set modes: G trains, D is frozen (acts as a static loss function)
        generator.train()
        discriminator.eval()

        # Sample new noise for G step
        z_g = mx.random.normal((batch_size, latent_dim))
        fake_labels_g = mx.random.randint(0, num_classes, (batch_size,))

        # Calculate loss and gradients for G
        g_loss, g_grads = g_val_and_grad_fn(z_g, fake_labels_g)

        # Update G's weights
        opt_g.update(generator, g_grads)

        # Eval to ensure ops complete
        mx.eval(discriminator.parameters(), generator.parameters())

        # ===================================================================
        # END: Training Step
        # ===================================================================

        losses["epoch"].append(epoch + 1)
        losses["batch"].append(epoch * num_batches + i)
        losses["d_loss"].append(float(d_loss))
        losses["g_loss"].append(float(g_loss))

    end_time_epoch = time.time()
    tqdm.write(
        f"Epoch {epoch+1}/{epochs} | D Loss: {float(d_loss):.4f} | G Loss: {float(g_loss):.4f} | Time: {end_time_epoch - start_time_epoch:.4f} seconds"
    )

end_time_training = time.time()
print(
    f"Training Complete! Time taken: {end_time_training - start_time_training:.4f} seconds"
)

# --- 7. Save Losses to File ---
import os

time_str = time.strftime("%Y%m%d-%H%M%S")
folder_path = f"models_{time_str}"

os.makedirs(folder_path, exist_ok=True)

# --- 7. Save Weights ---
generator.save_weights(f"{folder_path}/c_dcgan_generator_weights.npz")
discriminator.save_weights(f"{folder_path}/c_dcgan_discriminator_weights.npz")

np.savez(
    f"{folder_path}/gan_losses.npz",
    epoch=losses["epoch"],
    batch=losses["batch"],
    d_loss=losses["d_loss"],
    g_loss=losses["g_loss"],
)

# --- 8. Generate and Save Sample Images As Numpy ---

num_samples = 16

z_samples = mx.random.normal((num_samples, latent_dim))
label_samples = mx.random.randint(0, num_classes, (num_samples,))

start_time = time.time()

print(f"Generating {num_samples} images with labels: {label_samples}")

generated_images = generator(z_samples, label_samples)

end_time_generation = time.time()

mx.eval(generated_images)

end_time_eval = time.time()

print(f"Time taken for generation: {end_time_generation - start_time:.4f} seconds")
print(f"Time taken for eval: {end_time_eval - end_time_generation:.4f} seconds")

mx.savez(
    f"{folder_path}/c_dcgan_generated_images.npz",
    images=generated_images,
    labels=label_samples,
)
