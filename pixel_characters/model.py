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
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=5, stride=1, padding=0, bias=False),
            # the output of this layer is (batch_size, 512, 5, 5)
            # this is because of the formula for calculating the output size of the image
            # dilation is 1 by default
            # H_out = (H_in - 1) * stride - 2 * padding + dilation[0] * (kernel_size[0] - 1) + output_padding + 1
            #                      H_in(1),stride(1),pad(0),kernel(5),output_pad(0)
            # in here, 5 = (1–1) * 1 – 2 * 0 + 1 * (5 –1) + 0 + 1 ; 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # what is passed here is (batch_size, 512, 4, 4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            # this is because of the formula for calculating the output size of the image
            # H_out = (H_in - 1) * stride - 2 * padding + dilation[0] * (kernel_size[0] - 1) + output_padding + 1
            # H_out = (5    - 1) * 2      - 2 * 1       + 1           * (4              - 1) + 0              + 1
            # H_out = 4 * 2 - 2 + 3 + 1 = 10
            nn.BatchNorm2d(256),
            nn.ReLU(),
        

            # what is passed here is (batch_size, 256, 8, 8)
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            # the output of this layer is (batch_size, 128, 16, 16)
            # this is because of the formula for calculating the output size of the image
            # H_out = (H_in - 1) * stride - 2 * padding + dilation[0] * (kernel_size[0] - 1) + output_padding + 1
            # H_out = (10    - 1) * 2      - 2 * 1       + 1           * (4              - 1) + 0              + 1
            # H_out = 9 * 2 - 2 + 3 + 1 = 20
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            # input is (3, 20, 20)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            # output is (32, 10, 10)
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32*2),
            nn.LeakyReLU(0.2),
            # output is (64, 5, 5)

            nn.Conv2d(in_channels=64, out_channels=32*4, kernel_size=4, stride=2, padding=1, bias=False),
            # output is (1, 3, 3)
            nn.BatchNorm2d(32*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=32*4, out_channels=1, kernel_size=2, stride=2, padding=0, bias=False),
            # output is (1, 1, 1)
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)
    


