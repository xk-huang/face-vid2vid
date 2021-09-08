import torch
from torch import nn
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d, SynchronizedBatchNorm3d as BatchNorm3d
import torch.nn.functional as F
from modules.utils import ResBlock, DownBlock


class ApearanceEncoder(nn.Module):
    """
    a e
    inputs:
        x: (n, c, h, w)
    outputs:
        f_s: (n, block_expansion, d, h/num_down_blocks, w/num_down_blocks)
    """

    def __init__(self, depth, in_features=3, num_down_blocks=2, num_res_blocks=3, block_expansion=32):
        super(ApearanceEncoder, self).__init__()

        self.depth = depth
        self.block_expansion = block_expansion

        self.conv1 = nn.Conv2d(
            in_features, block_expansion * 2, kernel_size=7, padding=3)
        self.norm1 = BatchNorm2d(block_expansion * 2)

        downs = []
        for i in range(num_down_blocks):
            in_ch = block_expansion * (2 ** (i + 1))
            out_ch = block_expansion * (2 ** (i + 2))
            downs.append(DownBlock(in_ch, out_ch, use_3d=False))
        self.downs = nn.Sequential(*downs)

        in_ch = block_expansion * (2 ** (num_down_blocks + 1))
        out_ch = depth * block_expansion
        self.unsequeeze_conv = nn.Conv2d(in_ch, out_ch, 1)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(block_expansion, use_3d=True))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.in_features = in_features

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.downs(out)
        out = self.unsequeeze_conv(out)
        out = out.view(out.shape[0], self.block_expansion,
                       self.depth, *out.shape[-2:])

        return out
