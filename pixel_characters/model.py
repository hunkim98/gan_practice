import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # input to generator is something like (batch_size, 100, 1, 1)
        # this means that we are passing a batch of 32 100 random numbers to the generator
        self.model = nn.Sequential(
            # why is in_channels 100?
            # because we are passing 100 random numbers to the generator
            # our latent space is 100 dimensional!
            # first our input is (batch_size, 100, 1, 1)
            # step size is always 1!
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=5, stride=2, padding=0, bias=False),
            # the output of this layer is (batch_size, 512, 5, 5)
            # this is because of the formula for calculating the output size of the image
            # dilation is 1 by default
            # H_out = (H_in - 1) * stride - 2 * padding + dilation[0] * (kernel_size[0] - 1) + output_padding + 1
    #                     H_in(1),stride(2),pad(0),kernel(5),output_pad(0)
            # in here, 5 = (1–1) * 2 – 2 * 0 + 1 * (5 –1) + 0 + 1 = 5
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # what is passed here is (batch_size, 512, 4, 4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=0, bias=False),
            # this is because of the formula for calculating the output size of the image
            # H_out = (H_in - 1) * stride - 2 * padding + dilation[0] * (kernel_size[0] - 1) + output_padding + 1
            # H_out = (5    - 1) * 2      - 2 * 0       + 1           * (4              - 1) + 0              + 1
            # H_out = 4 * 2 - 0 + 3 + 1 = 12
            nn.BatchNorm2d(256),
            nn.ReLU(),
        

            # what is passed here is (batch_size, 256, 8, 8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=1, bias=False),
            # the output of this layer is (batch_size, 128, 16, 16)
            # this is because of the formula for calculating the output size of the image
            # H_out = (H_in - 1) * stride - 2 * padding + dilation[0] * (kernel_size[0] - 1) + output_padding + 1
            # H_out = (12    - 1) * 2      - 2 * 1       + 1           * (5              - 1) + 0              + 1
            # H_out = 11 * 2 - 2 + 4 + 1 = 25
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            # the output of this layer is (batch_size, 64, 32, 32)
            # this is because of the formula for calculating the output size of the image
            # H_out = (H_in - 1) * stride - 2 * padding + dilation[0] * (kernel_size[0] - 1) + output_padding + 1
            # H_out = (25    - 1) * 2      - 2 * 1       + 1           * (4              - 1) + 0              + 1
            # H_out = 24 * 2 - 2 + 3 + 1 = 50
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)



class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            # input is (3, 50, 50)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            # H_out = (50    + 2 * 1       - 1        * (4           - 1) - 1) / 2      + 1
            # H_out = (52   - 3           - 1) / 2 + 1
            # H_out = 25
            # output is (32, 25, 25)
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32*2),
            nn.LeakyReLU(0.2),
            # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            # H_out = (25    + 2 * 1       - 1        * (3           - 1) - 1) / 2      + 1
            # H_out = (27   - 2           - 1) / 2 + 1
            # H_out = 24 / 2 + 1 = 13
            # output is (64, 13, 13)

            nn.Conv2d(in_channels=64, out_channels=32*4, kernel_size=5, stride=2, padding=1, bias=False),
            # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            # H_out = (13    + 2 * 1       - 1        * (5           - 1) - 1) / 2      + 1
            # H_out = (15   - 4           - 1) / 2 + 1
            # H_out = 6
            nn.BatchNorm2d(32*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=32*4, out_channels=32*8, kernel_size=2, stride=2, padding=0, bias=False),
            # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            # H_out = (6    + 2 * 0       - 1        * (2           - 1) - 1) / 2      + 1
            # H_out = (6   - 1           - 1) / 2 + 1
            # H_out = 2
            nn.BatchNorm2d(32*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=32*8, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
            # H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
            # H_out = (2    + 2 * 0       - 1        * (2           - 1) - 1) / 1      + 1
            # H_out = (2   - 1           - 1) / 1 + 1
            # H_out = 1
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)
    




