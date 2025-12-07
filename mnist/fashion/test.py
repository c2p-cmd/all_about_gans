import mlx.core as mx
import mlx.nn as nn
import PIL.Image as Image
import numpy as np


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


decoder = Decoder(latent_dim=12)
decoder.load_weights("../../models_20251114-205441/decoder_weights_150.safetensors")
images = mx.random.uniform(0, 1, (16, 12))
reconstructed = decoder(images)
mx.eval(reconstructed)

# save reconstructed images to each file
for i in range(reconstructed.shape[0]):
    Image.fromarray(np.array(reconstructed[i].squeeze() * 255).astype(np.uint8)).save(
        f"reconstructed_{i}.png"
    )
