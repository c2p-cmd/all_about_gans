import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
# number of discriminator filters
ndf = 64
# number of color channels
nc = 3


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output


model = Generator(ngpu).to("mps")
model.load_state_dict(
    torch.load(
        "weights/netG_epoch_15.pth",
        map_location=torch.device("mps"),
    )
)
model.eval()

z = torch.randn(64, nz, 1, 1, device="mps")
fake_images = model(z).detach().cpu()

# Plot the generated images
grid_size = 8
fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
for i in range(grid_size):
    for j in range(grid_size):
        axes[i, j].imshow(
            np.transpose(
                (fake_images[i * grid_size + j] + 1) / 2, (1, 2, 0)
            )
        )
        axes[i, j].axis("off")
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
