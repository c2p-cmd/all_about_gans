import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial
import matplotlib.pyplot as plt
import os
from vae_silu import VAE


def load_cifar10():
    try:
        from keras.datasets import cifar10

        (X_train, _), (X_test, _) = cifar10.load_data()
    except ImportError:
        print("Error loading Keras dataset.")
        return None, None
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return mx.array(X_train), mx.array(X_test)


def loss_fn(model, X):
    X_recon, mu, logvar = model(X)
    # L1 Loss for sharpness
    recon_loss = mx.mean(mx.sum(mx.abs(X - X_recon), axis=(1, 2, 3)))
    kl_loss = mx.mean(
        -0.5 * mx.sum(1 + logvar - mx.square(mu) - mx.exp(logvar), axis=1)
    )

    # Low Beta to prioritize reconstruction
    beta = 0.001
    return recon_loss + (beta * kl_loss), (recon_loss, kl_loss)


def batch_iterate(X, batch_size):
    perm = mx.random.permutation(X.shape[0])
    for s in range(0, X.shape[0], batch_size):
        ids = perm[s : s + batch_size]
        yield mx.array(X[ids])


def check_reconstruction(model, X_test, folder_path, e):
    # Visualize REAL inputs vs RECONSTRUCTED outputs
    originals = X_test[:8]
    recons, _, _ = model(mx.array(originals))
    mx.eval(recons)
    originals = np.array(originals)
    recons = np.array(recons)

    plt.figure(figsize=(12, 4))
    for i in range(8):
        plt.subplot(2, 8, i + 1)
        plt.imshow(originals[i])
        plt.axis("off")
        plt.subplot(2, 8, i + 9)
        plt.imshow(recons[i])
        plt.axis("off")
    plt.savefig(f"{folder_path}/recon_epoch_{e+1}.png")
    plt.close()


def main(folder_path):
    BATCH_SIZE = 64
    EPOCHS = 20
    LATENT_DIM = 200

    print("Loading CIFAR-10...")
    X_train, X_test = load_cifar10()
    if X_train is None:
        return

    model = VAE(latent_dim=LATENT_DIM)
    model.load_weights("../models_20251120-204923/vae_50.npz")
    mx.eval(model.parameters())

    # LR Scheduler: High start (1e-3) -> Low end (1e-4)
    num_steps = EPOCHS * (len(X_train) // BATCH_SIZE)
    lr_schedule = optim.cosine_decay(1e-3, num_steps)
    optimizer = optim.Adam(learning_rate=lr_schedule)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(X):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        (loss, (recon, kl)), grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)
        return loss, (recon, kl)

    # ------------------------------------

    print(f"Starting training for {EPOCHS} epochs...")

    for e in range(EPOCHS):
        tic = time.time()
        total_recon = 0
        total_loss = 0
        total_kl = 0
        steps = 0
        model.train()

        for batch in batch_iterate(X_train, BATCH_SIZE):
            loss, (recon, kl) = train_step(batch)
            # Force evaluation of the STATE updates
            mx.eval(state)

            total_loss += loss.item()
            total_kl += kl.item()
            total_recon += recon.item()
            steps += 1

        toc = time.time()
        print(
            f"Epoch {e+1}: Loss {total_loss/steps:.2f} Recon Loss {total_recon/steps:.2f} KL Loss {total_kl/steps:.2f} - Time {toc - tic:.2f}s"
        )

        if (e + 1) % 10 == 0:
            model.eval()
            check_reconstruction(model, X_test, folder_path, e)
            model.save_weights(f"{folder_path}/vae_{e+1}.safetensors")

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(lr_schedule)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Total Loss")
    plt.plot(total_recon, label="Reconstruction Loss")
    plt.plot(total_kl, label="KL Divergence Loss")
    plt.title("Training Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(f"{folder_path}/lr_schedule.png")
    plt.show()


if __name__ == "__main__":
    time_str = time.strftime("%Y%m%d-%H%M%S")
    folder_path = f"../models_{time_str}"
    os.makedirs(folder_path, exist_ok=True)
    main(folder_path)
