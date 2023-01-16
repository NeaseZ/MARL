import numpy as np
import torch
from torch import nn

class FAB(nn.Module):
    def __init__(self, nc=2048, reduction=8):
        super().__init__()

        self.c0 = nn.Conv2d(nc, 1, kernel_size=1, padding=0)
        self.c1 = nn.Conv2d(nc, nc, kernel_size=1, padding=0)
        self.c2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, dilation=1)
        self.sm = nn.Softmax(dim=1)
        self.se = nn.Sequential(
            nn.Linear(nc, nc // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nc // reduction, nc, bias=False),
            nn.Sigmoid()
        )

        self.se1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.c0(x)
        x1 = self.sm(x0.flatten(2).permute(0, 2, 1))
        x2 = torch.matmul(x.flatten(2), x1)
        x3 = self.se(x2.flatten(1)).reshape()

        x4 = self.c1(x)
        x5 = torch.mul(x3, x4)

        x6 = self.c0(x5)#
        x7 = self.c2(x6)
        x8 = self.se1(x7)
        x9 = torch.mul(x6, x8)#

        x10 = x + x9

        return torch.tensor(x10)

if __name__ == '__main__':
    input = torch.randn(50, 512, 8, 8)
    se = FAB()
    output = se(input)
    # print(output.shape)
