import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from vae import VAE
import matplotlib.pyplot as plt
import os
import time
from functools import partial


def load_cifar10():
    # Trying to use keras just to fetch data easily as per your notebook context
    # You can replace this with any numpy array loader for CIFAR-10
    try:
        from keras.datasets import cifar10

        (X_train, _), (X_test, _) = cifar10.load_data()
    except ImportError:
        print(
            "Please ensure tensorflow/keras is installed to load CIFAR10 via this method, or replace with custom loader."
        )
        return None, None

    # Normalize to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    return X_train, X_test


def loss_fn(model, X):
    X_recon, mu, logvar = model(X)

    # Reconstruction Loss (MSE)
    # Sum over all pixels, abs over batch
    recon_loss = mx.sum(mx.abs(X - X_recon), axis=(1, 2, 3))
    recon_loss = mx.mean(recon_loss)

    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * mx.sum(1 + logvar - mx.square(mu) - mx.exp(logvar), axis=1)
    kl_loss = mx.mean(kl_loss)

    # Alpha weighting (as per common VAE implementations)
    beta = 1e-2
    return recon_loss + (beta * kl_loss), (recon_loss, kl_loss)


def batch_iterate(X, batch_size):
    perm = np.random.permutation(len(X))
    for s in range(0, len(X), batch_size):
        ids = perm[s : s + batch_size]
        yield mx.array(X[ids])


def gen_images(model: VAE, folder_path: str, e: int):
    images = model.decoder(mx.random.normal((16, 200)))
    mx.eval(images)
    images = np.array(images)

    plt.figure(figsize=(12, 10))
    for idx, image in enumerate(images, 1):
        plt.subplot(4, 4, idx)
        plt.imshow(image)
        plt.axis("off")
    plt.savefig(f"{folder_path}/images_{e+1}.png")
    plt.close()


def main(folder_path):
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 25
    LEARNING_RATE = 1e-4
    LATENT_DIM = 200
    fine_tune = True

    # Load Data
    print("Loading CIFAR-10...")
    X_train, X_test = load_cifar10()
    if X_train is None:
        return

    # Initialize Model
    model = VAE(latent_dim=LATENT_DIM)
    if fine_tune:
        model.load_weights("../models_20251120-183153/cifar10_vae_10.safetensors")
    mx.eval(model.parameters())

    # Optimizer
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    @partial(
        mx.compile,
        inputs=[model.state, optimizer.state],
        outputs=[model.state, optimizer.state],
    )
    def train_step(X):
        # State function for optimization
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        (loss, (recon, kl)), grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)
        return (loss, (recon, kl))

    print(f"Starting training for {EPOCHS} epochs...")

    for e in range(EPOCHS):
        tic = time.time()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        steps = 0
        model.train()

        for batch in batch_iterate(X_train, BATCH_SIZE):
            loss, (recon, kl) = train_step(batch)
            mx.eval([model.state, optimizer.state])

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            steps += 1

        toc = time.time()
        print(
            f"Epoch {e+1}: Loss {total_loss/steps:.2f} "
            f"(Recon {total_recon/steps:.2f}, KL {total_kl/steps:.2f}) "
            f"- Time {toc - tic:.2f}s"
        )
        if (e + 1) % 5 == 0:
            model.save_weights(f"{folder_path}/cifar10_vae_{e+1}.safetensors")
        if (e + 1) % 10 == 0:
            model.eval()
            gen_images(model, folder_path, e)

    # Save model
    model.save_weights(f"{folder_path}/cifar10_vae.safetensors")
    print("Model saved to cifar10_vae.safetensors")


if __name__ == "__main__":
    time_str = time.strftime("%Y%m%d-%H%M%S")
    folder_path = f"../models_{time_str}"
    os.makedirs(folder_path, exist_ok=True)
    main(folder_path)
