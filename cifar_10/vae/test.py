import mlx.core as mx
from vae_silu import VAE
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10

(_, _), (X_test, _) = cifar10.load_data()
test = (X_test[:8].astype("float32")) / 255.0
test = mx.array(test)
model = VAE(latent_dim=200)
model.load_weights("../models_20251120-213618/vae_10.safetensors")

mx.eval(model.parameters())

images, _, _ = model(test)
mx.eval(images)
images = np.array(images)

plt.figure(figsize=(8, 6))
for idx, image in enumerate(images, 1):
    plt.subplot(2, 8, idx)
    plt.imshow(image)
    plt.axis("off")
    if idx == 0:
        plt.title("Generated Images")

for idx, image in enumerate(test, 9):
    plt.subplot(2, 8, idx)
    plt.imshow(image)
    plt.axis("off")
    if idx == 8:
        plt.title("Real Images")
plt.show()
