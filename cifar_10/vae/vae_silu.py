import mlx.core as mx
import mlx.nn as nn


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class UpsamplingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
        )

    def __call__(self, x):
        x = upsample_nearest(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_dim=200):
        super().__init__()
        # Increased capacity: 64 -> 128 -> 256
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.flatten_size = 4 * 4 * 256
        self.output_layer = nn.Linear(self.flatten_size, latent_dim)
        self.bn_out = nn.BatchNorm(latent_dim)

    def __call__(self, x):
        x = nn.silu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.silu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.silu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.silu(self.bn_out(self.output_layer(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=200):
        super().__init__()
        self.dense = nn.Linear(latent_dim, 4 * 4 * 256)
        self.bn_dense = nn.BatchNorm(4 * 4 * 256)

        self.upconv1 = UpsamplingConv2d(256, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm(128)

        self.upconv2 = UpsamplingConv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm(64)

        self.upconv3 = UpsamplingConv2d(64, 64, 3, padding=1)  # Extra layer to smooth
        self.bn3 = nn.BatchNorm(64)

        self.final_conv = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def __call__(self, x):
        x = nn.silu(self.bn_dense(self.dense(x)))
        x = x.reshape(-1, 4, 4, 256)
        x = nn.silu(self.bn1(self.upconv1(x)))  
        x = nn.silu(self.bn2(self.upconv2(x)))
        x = nn.silu(self.bn3(self.upconv3(x)))
        x = mx.sigmoid(self.final_conv(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=200):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.proj_mu = nn.Linear(latent_dim, latent_dim)
        self.proj_logvar = nn.Linear(latent_dim, latent_dim)

    def __call__(self, x):
        features = self.encoder(x)
        mu = self.proj_mu(features)
        logvar = self.proj_logvar(features)
        sigma = mx.exp(0.5 * logvar)
        eps = mx.random.normal(sigma.shape)
        z = mu + sigma * eps
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
