import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=in_plane,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_plane)
        self.point_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=out_plane,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

if __name__ == "__main__":
    input = torch.randn((8, 3, 32, 32))
    model = DWConv(3, 64)
    output = model(input)
    print(input.size(), '\n', output.size())

