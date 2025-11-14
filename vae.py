import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import time

time_str = time.strftime("%Y%m%d-%H%M%S")
folder_path = f"models_{time_str}"
os.makedirs(folder_path, exist_ok=True)

mx.random.seed(1337)


def sampling(z_mean, z_log_var) -> mx.array:
    """
    Reparameterization trick to sample from N(z_mean, exp(z_log_var))
    """
    batch = z_mean.shape[0]
    dim = z_mean.shape[1]
    epsilon = mx.random.normal(shape=(batch, dim))
    return z_mean + mx.exp(0.5 * z_log_var) * epsilon


latent_dim = 12
input_shape = (28, 28, 1)
batch_size = 128
epochs = 50


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.fc = nn.Linear(
            input_dims=7 * 7 * 64,
            output_dims=16,
        )
        self.z_mean = nn.Linear(
            input_dims=16,
            output_dims=latent_dim,
        )
        self.z_log_var = nn.Linear(
            input_dims=16,
            output_dims=latent_dim,
        )

    def __call__(self, x) -> tuple[mx.array, mx.array, mx.array]:
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = mx.reshape(x, [x.shape[0], -1])  # flatten
        x = nn.relu(self.fc(x))
        mu = self.z_mean(x)
        log_var = self.z_log_var(x)
        z = sampling(z_mean=mu, z_log_var=log_var)
        return mu, log_var, z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dims=latent_dim, output_dims=7 * 7 * 64)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.output = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x) -> mx.array:
        batch_size = x.shape[0]
        x = nn.relu(self.fc(x))
        x = mx.reshape(x, [batch_size, 7, 7, 64])
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.sigmoid(self.output(x))
        return x


def reconstruction_loss(inputs, targets):
    bce_loss = nn.losses.binary_cross_entropy(inputs, targets, reduction="none")
    return mx.mean(mx.sum(bce_loss, axis=(1, 2, 3)))


def kl_loss(z_mean, z_log_var):
    loss = -0.5 * (1 + z_log_var - mx.square(z_mean) - mx.exp(z_log_var))
    loss = mx.mean(mx.sum(loss, axis=1))
    return loss


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def __call__(self, x) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # Pass through encoder
        mu, log_var, z = self.encoder(x)
        # Pass through decoder
        recon = self.decoder(z)
        return mu, log_var, z, recon


def vae_loss_fn(model: VAE, batch: mx.array, beta: float):
    # Run the full model
    mu, log_var, _, recon = model(batch)

    # Calculate the two loss components
    rec_loss = reconstruction_loss(batch, recon)
    kl = kl_loss(mu, log_var)

    # Return the single, combined loss
    return rec_loss + beta * kl


vae_model = VAE(latent_dim=latent_dim)
print("VAE")
print(vae_model)
print()

mx.eval(vae_model.parameters())

val_and_grad_fn = nn.value_and_grad(vae_model, vae_loss_fn)

# Load MNIST
(X_train, _), (X_test, _) = fashion_mnist.load_data()
X_train = np.concatenate([X_train, X_test], axis=0)
# Normalize to [0, 1]
X_train = (X_train.astype(np.float32)) / 255
# Add a channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
X_train = np.expand_dims(
    X_train,
    axis=-1,
)
# Convert to MLX array
X_train = mx.array(X_train)

print(f"Training data shape: {X_train.shape}")

num_batches = len(X_train) // batch_size
losses = {
    "iterations": [],
    "total_loss": [],
    "lr": [],
}
kl_warmup_epochs = 20
beta = 0.0
optimizer = optim.Adam(
    learning_rate=optim.step_decay(
        init=1e-3,
        decay_rate=0.9,
        step_size=num_batches * 5,
    )
)

print("Starting Training...")
for epoch in range(epochs):
    beta = min(1.0, (epoch + 1) / kl_warmup_epochs)
    for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        # Sample a batch of real images
        idx = mx.random.randint(0, len(X_train), (batch_size,))
        real_images = X_train[idx]

        # Set model to training mode
        vae_model.train()

        # --- This is the entire update step ---
        # 1. Calculate loss and gradients for the *entire* VAE
        total_loss, grads = val_and_grad_fn(vae_model, real_images, beta)

        # 2. Update all parameters (both enc and dec) in one go
        optimizer.update(vae_model, grads)

        # 3. Evaluate all parameters
        mx.eval(vae_model.parameters())
        # --- End of update step ---

        # Logging (adjust variable names)
        losses["iterations"].append(int(epoch * num_batches + i))
        losses["total_loss"].append(float(total_loss))
        losses["lr"].append(float(optimizer.learning_rate))

    tqdm.write(
        f"Epoch: {epoch+1}/{epochs}, Total Loss: {float(total_loss):.4f}, LR: {float(optimizer.learning_rate):.6f}, Beta: {float(beta):.6f}"
    )
    if (epoch + 1) % 5 == 0:
        # Save weights from the combined model
        vae_model.save_weights(f"{folder_path}/vae_weights_{epoch+1}.safetensors")
        vae_model.encoder.save_weights(
            f"{folder_path}/encoder_weights_{epoch+1}.safetensors"
        )
        vae_model.decoder.save_weights(
            f"{folder_path}/decoder_weights_{epoch+1}.safetensors"
        )
        # sample some images
        vae_model.eval()

        _samples = mx.random.normal((4, latent_dim))
        generated_images = vae_model.decoder(_samples)
        mx.eval(generated_images)
        generated_images = np.array(generated_images).squeeze()
        try:
            plt.figure(figsize=(8, 1))
            for _i in [1, 2, 3, 4]:
                plt.subplot(1, 4, _i)
                plt.imshow(generated_images[_i - 1], cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"{folder_path}/vae_image_grid.png")
            plt.close("all")
        except Exception as e:
            print("Error saving images", e)
    if (epoch + 1) == epochs:
        # Save weights from the combined model
        vae_model.save_weights(f"{folder_path}/vae_weights.safetensors")
        vae_model.encoder.save_weights(f"{folder_path}/encoder_weights.safetensors")
        vae_model.decoder.save_weights(f"{folder_path}/decoder_weights.safetensors")


vae_model.eval()

try:
    mx.eval(losses["total_loss"], losses["lr"])
    pd.DataFrame(
        {
            "Iterations": losses["iterations"],
            "Total Loss": losses["total_loss"],
            "Learning Rate": losses["lr"],
        }
    ).to_csv(f"{folder_path}/losses.csv")
except Exception as e:
    print(f"Error while saving losses ", e)


def plot_label_clusters(encoder, data, labels):
    from sklearn.manifold import TSNE

    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder(data)
    mx.eval(z_mean)

    # We have a 12D space, so project it down to 2D using t-SNE
    print("Running t-SNE on latent space...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    z_mean_2d = tsne.fit_transform(
        np.array(z_mean)
    )  # Convert MLX array to numpy for sklearn
    print("t-SNE finished.")

    plt.figure(figsize=(12, 8))
    # Plot the new 2D coordinates
    plt.scatter(z_mean_2d[:, 0], z_mean_2d[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.savefig(f"{folder_path}/latent_space_tsne.png")
    # plt.show() # In this environment, plt.show() will fail. Use savefig only.
    plt.close("all")  # Add this to free up memory


(x_train, y_train), _ = fashion_mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
x_train = mx.array(x_train)

plot_label_clusters(vae_model.encoder, x_train, y_train)
