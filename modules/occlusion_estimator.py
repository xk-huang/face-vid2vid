import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import UNetEncoder, UNetDecoder, get_multi_sample_grid, warp_multi_feature_volume


class OcclusionEstimator(nn.Module):
    r"""
    flow occlusion estimator
    input:
        features (n, c, d, h, w)
        source_keypoint (n, num_kp, 3)
        target_keypoint (n, num_kp, 3)
        source_rot (n, 3, 3)
        target_rot (n, 3, 3)
    intermediate network input:
        (num_kp + 1) grid (n, num_kp + 1, d, h, w, 3)
        (num_kp + 1) warped-features, 1 means no warping (n, num_kp + 1, c, d, h, w)
    output:
        3d flow mask: (n, num_kp + 1, d, h, w), softmax
        2d occlusion mask: (n, 1, h , w), sigmoid
    """

    def __init__(self, num_features_ch=32, compressed_in_features=5, block_expansion=32, num_blocks=5, max_features=1024, num_kp=20, depth=16, use_skip=False) -> None:
        super(OcclusionEstimator, self).__init__()

        self.num_kp = num_kp
        self.depth = depth
        self.in_features = compressed_in_features * (num_kp + 1)
        self.compressed_in_features = compressed_in_features

        self.compress_input = nn.Conv3d(
            num_features_ch, compressed_in_features, 1)

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

        self.flow_3d_mask_layer = nn.Conv3d(
            self.unet_out_ch, num_kp + 1, 7, padding=3)
        self.feature_2d_mask_layer = nn.Conv2d(
            depth * self.unet_out_ch, 1, 7, padding=3)

    def forward(self, features, source_keypoint, target_keypoint, source_rot, target_rot):
        compressed_features = self.compress_input(features)
        n, c, d, h, w = compressed_features.shape
        grids = get_multi_sample_grid(
            compressed_features, source_keypoint, target_keypoint, source_rot, target_rot)
        warped_features = warp_multi_feature_volume(compressed_features, grids)
        warped_features = warped_features.reshape(n, -1, d, h, w)
        print("[test] start warping")
        out_features = self.encoder(warped_features)
        out_features = self.decoder(out_features)
        cat_features = torch.cat([warped_features, out_features], dim=1)

        flow_3d_mask = self.flow_3d_mask_layer(cat_features)
        num_kp = flow_3d_mask.shape[1]
        flow_3d_mask = flow_3d_mask.view(n, num_kp, -1)
        flow_3d_mask = F.softmax(flow_3d_mask, -1)
        flow_3d_mask = flow_3d_mask.view(n, num_kp, d, h, w)

        feature_2d_mask = self.feature_2d_mask_layer(
            cat_features.view(n, -1, h, w))
        feature_2d_mask = torch.sigmoid(feature_2d_mask)

        return {
            "flow_3d_mask": flow_3d_mask,
            "feature_2d_mask": feature_2d_mask
        }
