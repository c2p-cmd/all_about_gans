import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib.pyplot as plt
import os
from functools import partial
from tqdm import tqdm


# --- 1. Utilities ---
def load_cifar10():
    try:
        from keras.datasets import cifar10

        (X_train, _), (_, _) = cifar10.load_data()
    except ImportError:
        print("Please ensure tensorflow/keras is installed or provide custom loader.")
        return None

    # Normalize to [-1, 1] for GANs (Standard practice)
    X_train = (X_train.astype("float32") - 127.5) / 127.5
    return mx.array(X_train)


def upsample_nearest(x, scale=2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class UpsamplingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding
        )

    def __call__(self, x):
        return self.conv(upsample_nearest(x))


# --- 2. Models (Adapted for 32x32) ---


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        # Project and Reshape: 100 -> 4x4x256
        self.linear = nn.Linear(latent_dim, 4 * 4 * 256)
        self.bn0 = nn.BatchNorm(4 * 4 * 256)

        # 4x4 -> 8x8
        self.up1 = UpsamplingConv2d(256, 128)
        self.bn1 = nn.BatchNorm(128)

        # 8x8 -> 16x16
        self.up2 = UpsamplingConv2d(128, 64)
        self.bn2 = nn.BatchNorm(64)

        # 16x16 -> 32x32
        self.up3 = UpsamplingConv2d(64, 3)
        # No Batch Norm on final layer for Generator

    def __call__(self, x):
        # Dense Block
        x = nn.relu(self.bn0(self.linear(x)))
        x = x.reshape(-1, 4, 4, 256)

        # Conv Blocks
        x = nn.relu(self.bn1(self.up1(x)))
        x = nn.relu(self.bn2(self.up2(x)))

        # Output: Tanh for [-1, 1] range
        x = mx.tanh(self.up3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 32x32 -> 16x16
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        # No Batch Norm on first layer of Discriminator

        # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm(128)

        # 8x8 -> 4x4
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm(256)

        # Flatten and Output Probability
        self.flatten = nn.Linear(4 * 4 * 256, 1)

    def __call__(self, x):
        x = nn.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = nn.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = nn.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)

        x = x.reshape(x.shape[0], -1)
        x = self.flatten(x)
        # Return logits (Sigmoid handled in loss function usually, or manually)
        return x


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


# --- 3. Training ---


def train_gan():
    # Hyperparams
    BATCH_SIZE = 64
    EPOCHS = 20
    LR_D = 1e-5  # DCGAN standard LR
    LR_G = 3e-5  # DCGAN standard LR
    LATENT_DIM = 100

    os.makedirs("output_gan", exist_ok=True)

    print("Loading Data...")
    X_train = load_cifar10()
    if X_train is None:
        return

    # Initialize
    generator = Generator(LATENT_DIM)
    generator.apply(init_conv_weight, is_conv_weight)
    generator.apply(init_bn_weight, is_bn_weight)
    generator.load_weights("output_gan tuned2/generator_epoch_20.safetensors")

    discriminator = Discriminator()
    discriminator.apply(init_conv_weight, is_conv_weight)
    discriminator.apply(init_bn_weight, is_bn_weight)
    discriminator.load_weights("output_gan tuned2/discriminator_epoch_20.safetensors")

    mx.eval(generator.parameters(), discriminator.parameters())

    # Optimizers
    opt_g = optim.Adam(learning_rate=LR_G, betas=[0.5, 0.999])
    opt_d = optim.Adam(learning_rate=LR_D, betas=[0.5, 0.999])

    def get_noise_std(epoch):
        # Linear decay from 0.1 to 0.0
        return max(0.0, 0.1 * (1 - epoch / EPOCHS))

    # Loss Function (Binary Cross Entropy with Logits)
    def d_loss_fn(d_model, g_model, real_images, z, noise_std):
        fake_images = g_model(z)

        noise = mx.random.normal(real_images.shape) * noise_std

        real_logits = d_model(real_images + noise)
        fake_logits = d_model(fake_images + noise)

        # Real Label = 1, Fake Label = 0
        real_labels = mx.random.uniform(0.75, 1.0, real_logits.shape)
        real_loss = mx.mean(
            nn.losses.binary_cross_entropy(
                real_logits,
                real_labels,
                with_logits=True,
            )
        )
        fake_loss = mx.mean(
            nn.losses.binary_cross_entropy(
                fake_logits,
                mx.ones_like(fake_logits) * 0.1,  # Label smoothing for fake images
                with_logits=True,
            )
        )

        return real_loss + fake_loss

    def g_loss_fn(d_model, g_model, z):
        fake_images = g_model(z)
        fake_logits = d_model(fake_images)

        # Generator wants Discriminator to think images are Real (Label = 1)
        return mx.mean(
            nn.losses.binary_cross_entropy(
                fake_logits,
                mx.ones_like(fake_logits),
                with_logits=True,
            )
        )

    # State updates
    d_state = [discriminator.state, opt_d.state]
    g_state = [generator.state, opt_g.state]

    # Step Functions
    def train_step_d(real_images, noise_std):
        z = mx.random.normal((real_images.shape[0], LATENT_DIM))
        loss, grads = nn.value_and_grad(
            discriminator,
            lambda d: d_loss_fn(d, generator, real_images, z, noise_std),
        )(discriminator)
        opt_d.update(discriminator, grads)
        return loss

    def train_step_g(batch_size):
        z = mx.random.normal((batch_size, LATENT_DIM))
        loss, grads = nn.value_and_grad(
            generator, lambda g: g_loss_fn(discriminator, g, z)
        )(generator)
        opt_g.update(generator, grads)
        return loss

    print("Starting Training...")

    losses = {"d_loss": [], "g_loss": []}

    for e in range(EPOCHS):
        tic = time.time()
        generator.train()
        discriminator.train()

        d_losses = []
        g_losses = []

        current_noise = get_noise_std(e)

        # Batch Iterate
        perm = mx.random.permutation(X_train.shape[0])
        for i in tqdm(
            range(0, X_train.shape[0], BATCH_SIZE),
            desc=f"Epoch {e+1}/{EPOCHS}",
            unit="batch",
        ):
            batch_idx = perm[i : i + BATCH_SIZE]
            real_images = X_train[batch_idx]

            # Train Discriminator
            l_d = train_step_d(real_images, current_noise)
            mx.eval(d_state)

            # Train Generator
            l_g = train_step_g(len(batch_idx))
            mx.eval(g_state)

            l_g = train_step_g(len(batch_idx))
            mx.eval(g_state)

        d_losses.append(l_d.item())
        g_losses.append(l_g.item())
        toc = time.time()
        tqdm.write(
            f"Epoch {e+1}: D_Loss {np.mean(d_losses):.4f} G_Loss {np.mean(g_losses):.4f} - {toc-tic:.2f}s"
        )
        losses["d_loss"].append(np.mean(d_losses))
        losses["g_loss"].append(np.mean(g_losses))

        if (e + 1) % 5 == 0:
            # Save model weights
            generator.save_weights(
                f"output_gan/generator_epoch_{e+1}.safetensors",
            )
            discriminator.save_weights(
                f"output_gan/discriminator_epoch_{e+1}.safetensors",
            )

        # Save Images
        if (e + 1) % 10 == 0:
            generator.eval()
            z = mx.random.normal((16, LATENT_DIM))
            imgs = generator(z)
            mx.eval(imgs)
            imgs = np.array(imgs)

            # Denormalize [-1, 1] -> [0, 1]
            imgs = (imgs + 1) / 2.0

            plt.figure(figsize=(8, 8))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.imshow(imgs[i])
                plt.axis("off")
            plt.savefig(f"output_gan/epoch_{e+1}.png")
            plt.close()
            generator.train()

    plt.figure(figsize=(10, 5))
    plt.plot(losses["d_loss"], label="Discriminator Loss")
    plt.plot(losses["g_loss"], label="Generator Loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("output_gan/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    train_gan()
