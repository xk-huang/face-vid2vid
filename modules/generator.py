import pdb
from numpy.lib.utils import source
import torch
from torch.functional import align_tensors
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import ResBlock, UpBlock, get_multi_sample_grid, warp_multi_feature_volume
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class OcclAwareGenerator(nn.Module):
    """
    Occlusion-aware generator
    inputs:
        features: (n, c, d, h, w)
        source_keypoint: (n, num_kp, 3)
        target_keypoint: (n, num_kp, 3)
        source_rot: (n, 3, 3)
        target_rot: (n, 3, 3)
        flow_3d_mask: (n, num_kp + 1, d, h, w), softmax
        feature_2d_mask: (n, 1, h , w), sigmoid
    outputs:
        rgb: (n, 3, h, w) re-scale to (0, 1) after tanh (-1, 1)
    """

    def __init__(self, depth=16, num_features_ch=32, num_res_blocks=6, num_up_blocks=2, num_kp=20, block_expansion=64, max_features=1024, sn=False) -> None:
        super(OcclAwareGenerator, self).__init__()

        self.num_kp = num_kp
        self.depth = depth

        interm_ch = min(block_expansion * (num_up_blocks ** 2), max_features)
        self.conv1 = nn.Conv2d(depth * num_features_ch,
                               interm_ch, 3, padding=1)
        if sn:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.norm1 = BatchNorm2d(interm_ch)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(interm_ch, interm_ch, 1)
        if sn:
            self.conv2 = nn.utils.spectral_norm(self.conv2)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(interm_ch, sn=sn))
        self.res_blocks = nn.Sequential(*res_blocks)

        up_blocks = []
        in_ch = interm_ch
        for i in range(num_up_blocks)[::-1]:
            out_ch = block_expansion * (2 ** i)
            up_blocks.append(UpBlock(in_ch, out_ch, sn=sn))
            in_ch = out_ch
        self.up_blocks = nn.Sequential(*up_blocks)

        self.rgb = nn.Conv2d(in_ch, 3, 7, padding=3)
        if sn:
            self.rgb = nn.utils.spectral_norm(self.rgb)

    def forward(self, features, source_keypoint, target_keypoint, source_rot, target_rot, flow_3d_mask, feature_2d_mask):
        # grids (n, num_kp + 1, d, h, w, 3)
        grids = get_multi_sample_grid(
            features, source_keypoint, target_keypoint, source_rot, target_rot)
        grids = grids.permute(0, 1, 5, 2, 3, 4)  # (n, num_kp + 1, 3, d, h, w)
        # flow_3d_mask (n, num_kp + 1, d, h, w)
        flow_3d_mask = flow_3d_mask.unsqueeze(2)  # (n, num_kp + 1, 1, d, h, w)
        grids = (grids * flow_3d_mask).sum(dim=1)  # (n, 3, d, h, w)
        grids = grids.permute(0, 2, 3, 4, 1)  # (n, d, h, w, 3)

        warped_features = F.grid_sample(features, grids, align_corners=True)
        warped_features = warped_features.view(
            warped_features.shape[0], -1, *warped_features.shape[-2:])

        out = self.conv1(warped_features)
        out = self.norm1(out)
        out = self.lrelu1(out)
        out = self.conv2(out)

        out = out * feature_2d_mask

        out = self.res_blocks(out)
        out = self.up_blocks(out)

        out = self.rgb(out)
        out = torch.tanh(out)
        out = (out + 1.0) / 2.0

        return out
