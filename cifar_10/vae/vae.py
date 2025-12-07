import mlx.core as mx
import mlx.nn as nn


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class UpsamplingConv2d(nn.Module):
    """
    Approximates Conv2DTranspose by upsampling followed by a convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        # Standard convolution with stride 1 to preserve the upsampled size
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding
        )

    def __call__(self, x):
        x = upsample_nearest(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        # Block 1: 3 -> 64
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Block 2: 64 -> 128
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Block 3: 128 -> 256
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4

        # Flatten and Dense
        self.flatten_size = 4 * 4 * 256
        self.output_layer = nn.Linear(self.flatten_size, latent_dim)
        self.bn_out = nn.BatchNorm(latent_dim)

    def __call__(self, x):
        # Block 1
        x = nn.relu(self.bn1_1(self.conv1_1(x)))
        x = nn.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # Block 2
        x = nn.relu(self.bn2_1(self.conv2_1(x)))
        x = nn.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # Block 3
        x = nn.relu(self.bn3_1(self.conv3_1(x)))
        x = nn.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        # Flatten and Output
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.bn_out(self.output_layer(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        # Dense layers to scale up
        self.dense1 = nn.Linear(latent_dim, 100)
        self.bn1 = nn.BatchNorm(100)

        self.dense2 = nn.Linear(100, 1024)
        self.bn2 = nn.BatchNorm(1024)

        self.dense3 = nn.Linear(1024, 4 * 4 * 256)
        self.bn3 = nn.BatchNorm(4 * 4 * 256)

        # Transposed Conv Block 1: 4x4 -> 8x8
        # Replaces Conv2DTranspose(256, 3, strides=2)
        self.upconv1 = UpsamplingConv2d(256, 256, 3, padding=1)
        self.bn_up1 = nn.BatchNorm(256)

        # Replaces Conv2DTranspose(256, 3, strides=1)
        self.conv1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn_conv1 = nn.BatchNorm(256)

        # Transposed Conv Block 2: 8x8 -> 16x16
        self.upconv2 = UpsamplingConv2d(256, 128, 3, padding=1)
        self.bn_up2 = nn.BatchNorm(128)

        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn_conv2 = nn.BatchNorm(128)

        # Transposed Conv Block 3: 16x16 -> 32x32
        self.upconv3 = UpsamplingConv2d(128, 64, 3, padding=1)
        self.bn_up3 = nn.BatchNorm(64)

        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_conv3 = nn.BatchNorm(64)

        # Final Layer to get 3 channels (RGB)
        self.final_conv = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def __call__(self, x):
        # Dense Expansions
        x = nn.relu(self.bn1(self.dense1(x)))
        x = nn.relu(self.bn2(self.dense2(x)))
        x = nn.relu(self.bn3(self.dense3(x)))

        # Reshape Bx(4*4*256) -> Bx4x4x256
        x = x.reshape(-1, 4, 4, 256)

        # Block 1 (4->8)
        x = nn.relu(self.bn_up1(self.upconv1(x)))
        x = nn.relu(self.bn_conv1(self.conv1(x)))

        # Block 2 (8->16)
        x = nn.relu(self.bn_up2(self.upconv2(x)))
        x = nn.relu(self.bn_conv2(self.conv2(x)))

        # Block 3 (16->32)
        x = nn.relu(self.bn_up3(self.upconv3(x)))
        x = nn.relu(self.bn_conv3(self.conv3(x)))

        # Final
        x = mx.sigmoid(self.final_conv(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        # Projections for Mu and LogVar
        self.proj_mu = nn.Linear(latent_dim, latent_dim)
        self.proj_logvar = nn.Linear(latent_dim, latent_dim)

    def __call__(self, x):
        # Encode
        features = self.encoder(x)
        mu = self.proj_mu(features)
        logvar = self.proj_logvar(features)

        # Reparameterize
        sigma = mx.exp(0.5 * logvar)
        eps = mx.random.normal(sigma.shape)
        z = mu + sigma * eps

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar
