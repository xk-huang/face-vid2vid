from ssl import HAS_ECDH
import torch
import torch.nn as nn
from modules.util import DownBlock, UpBlock, UNetEncoder, UNetDecoder


class OcclusionEstimator(nn.Module):
    """
    flow occlusion estimator
    input:
        features (n, c, d, h, w)
        source_keypoint
        target_keypoint
        source_rot
        target_rot
    intermediate network input:
        (num_kp + 1) grid (n, num_kp + 1, d, h, w, 3)
        (num_kp + 1) warped-features, 1 means no warping (n, num_kp + 1, c, d, h, w)
    output:
        3d flow mask: (n, num_kp + 1, d, h, w), softmax
        2d occlusion mask: (n, 1, h , w), sigmoid
    """

    def __init__(self, compressed_in_features=5, block_expansion=32, num_blocks=5, max_features=1024, num_kp=20, depth=16, use_skip=False) -> None:
        super(OcclusionEstimator, self).__init__()

        self.num_kp = num_kp
        self.depth = depth
        self.in_features = compressed_in_features * (num_kp + 1)
        self.compressed_in_features = compressed_in_features

        self.compress_input = nn.Conv3d(
            self.in_features, compressed_in_features, 1)

        self.encoder = UNetEncoder(
            True, block_expansion, self.in_features, num_blocks, max_features)
        self.decoder = UNetDecoder(
            True, block_expansion, self.in_features, num_blocks, max_features, use_skip=use_skip)

        self.unet_out_ch = self.decoder.num_out_ch + \
            (num_kp + 1) * compressed_in_features

        # both input feature volume is warped num_kp times, + itself with no warping
        # feature volumes:
        # before warping (N, in_ch, D, H, W)
        # warp grid (N, num_kp, d, h, w, 3)
        # after consequtive warping (N, num_kp + 1, in_ch, D, H, W)
        # heatmap (N, num_kp + 1, D, H, W), return it
        # final feature should be (N, in_ch, D, H, W)
        # need a warper func (N, in_ch, D, H, W) -> (N, num_kp + 1, in_ch, D, H, W)

        self.flow_3d_mask_layer = nn.Conv3d(self.unet_out_ch, num_kp + 1)
        self.feature_2d_mask_layer = nn.Conv2d(
            depth * self.unet_out_ch, 1, 7, padding=3)

    def forward(self, x):
        pass
