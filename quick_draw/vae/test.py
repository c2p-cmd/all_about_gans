from cvae import CVAE
from dataset import categories
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

model = CVAE(latent_dim=32, max_filters=64, num_classes=len(categories))
model.load_weights("./model_weights_latent_32_filters_64_ft2/cvae_epoch_120.safetensors")
model.eval()

labels = mx.arange(len(categories))

print("Generating samples...")
samples = model.sample(num_samples=len(categories) * 5, labels=mx.repeat(labels, 5))

samples = samples * 255.0
samples_np = np.array(samples, dtype=np.uint8)

grid_rows = len(categories)
grid_cols = 5
grid_img = np.zeros((grid_rows * 28, grid_cols * 28), dtype=np.uint8)
for i in range(grid_rows):
    for j in range(grid_cols):
        grid_img[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = samples_np[i * 5 + j].reshape(28, 28)

plt.imshow(grid_img, cmap="gray")
plt.axis("off")
plt.show()