from cvae import CVAE
from dataset import categories
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time

model = CVAE(latent_dim=32, max_filters=64, num_classes=len(categories))
model.load_weights(
    "./model_weights_latent_32_filters_64_ft2/cvae_epoch_120.safetensors"
)
model.eval()

warmup_iterations = 2

sample_count = 5
labels = mx.random.randint(0, len(categories), (sample_count,))
print("Generating samples...")
start_time = time.time()
for _ in range(warmup_iterations + 1):
    samples = model.sample(num_samples=sample_count, labels=labels)
end_time = time.time()
print(
    f"Time taken to generate {sample_count} samples: {end_time - start_time:.4f} seconds"
)
print("Sample generation complete.", samples.shape)

samples = samples * 255.0
samples_np = np.array(samples, dtype=np.uint8)

fig, axes = plt.subplots(1, sample_count, figsize=(15, 3))
for i in range(sample_count):
    axes[i].imshow(samples_np[i], cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(f"Label: {labels[i].item()}, {categories[int(labels[i].item())]}")
plt.suptitle("Generated Samples from CVAE", fontsize=16)
plt.tight_layout()
plt.show()
