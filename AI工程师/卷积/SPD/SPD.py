# code: https://github.com/LabSAINT/SPD-Conv

import torch
import torch.nn as nn


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

if __name__ == "__main__":

    # 输入和输出
    block = space_to_depth()
    input = torch.rand(3, 128, 64, 64)
    output = block(input)
    print(input.size(), '\n', output.size())