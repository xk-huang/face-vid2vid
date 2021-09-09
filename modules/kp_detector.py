#  3d keypoint estimator
#      1. facial canonical keypoint estimator
#      2. Head pose & exp deformation estimator
from torch import feature_dropout, nn
import torch
import torch.nn.functional as F

from modules.utils import UNetDecoder, UNetEncoder, AntiAliasInterpolation2d, make_coordinate_grid


class CanonicalKPDetector(nn.Module):
    """
    2d image to a set of keypoint
    input:
        - x: (n,c,h,w)
    output:
        - out: dict{
            "keypoint": kp: (n, num_kp, 3)
            (optional) "jacobian": jacobian: (n, num_kp, 2, 2) or (n, num_kp, 3, 3)
        }



    """

    def __init__(self, block_expansion, num_kp, num_in_channels, max_features, num_blocks, temperature, scale_factor=1, padding=0, depth=16):
        super(CanonicalKPDetector, self).__init__()

        self.depth = depth
        self.encoder = UNetEncoder(
            False, block_expansion, num_in_channels, num_blocks, max_features)
        self.unsqueeze_conv = nn.Conv2d(
            block_expansion * (2 ** num_blocks), block_expansion * (2 ** num_blocks) * depth, 1)
        self.decoder = UNetDecoder(
            True, block_expansion, num_in_channels, num_blocks, max_features, use_skip=False)

        self.kp = nn.Conv3d(
            self.decoder.num_out_ch, num_kp, 7, padding=padding)
        self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_in_channels, scale_factor)

    def gaussian2kp(self, heatmap):
        shape = heatmap.shape  # N,C,D,H,W
        # print(f"[test] heatmap shape {shape}")

        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(
            shape[2:], dtype=heatmap.type()).unsqueeze_(0).unsqueeze_(0).to(device=heatmap.device)
        # print(f"[test] grid shape {grid.shape}")
        keypoint = (
            heatmap * grid).sum(dim=tuple(range(2, len(grid.shape) - 1)))

        return keypoint

    def forward(self, x):
        if self.scale_factor != 1:
            # print(f"[test] scale {self.scale_factor}")
            x = self.down(x)

        in_shape = x.shape
        feature_map = self.encoder(x)[-1]
        # import pdb
        # pdb.set_trace()

        feature_map = self.unsqueeze_conv(feature_map)
        feature_map = feature_map.view(
            feature_map.shape[0], -1, self.depth, *feature_map.shape[-2:])
        feature_map = self.decoder(feature_map)

        heatmap = self.kp(feature_map)
        heatmap_shape = heatmap.shape
        heatmap = heatmap.view(*heatmap_shape[:2], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=-1)
        heatmap = heatmap.view(heatmap_shape)

        out = {'keypoint': self.gaussian2kp(heatmap)}
        if self.jacobian:
            raise NotImplementedError

        return out
