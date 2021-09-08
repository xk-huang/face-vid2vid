from torch import nn
import torch.nn.functional as F
from modules.utils import get_multi_sample_grid
import torch


class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    input:
        x: (n, c, h, w), range from 0 - 1, re-scaled after tanh
    outputs:
        feature_maps: list of intermediate features [(n, c_i, h_i, w_i) for i in this_list]
        predition_map: (n, 1, h_I, w_I)
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            # down_blocks.append(
            #     DownBlock2d(num_channels + num_kp * use_kp if i == 0 else min(max_features, block_expansion * (2 ** i)),
            #                 min(max_features, block_expansion * (2 ** (i + 1))),
            #                 norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))
            down_blocks.append(
                DownBlock2d(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(
            self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        # self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        # if self.use_kp:
        #     heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
        #     out = torch.cat([out, heatmap], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    inputs:
        x: (n, c, h, w)
    outputs:
        out_dict: {
            "feature_map_[scale]": feature_maps: list [(n, c_i, h_i, w_i) for i in this_list]
            "prediction_map_[scale]": prediction_map: (n, 1, h_I, w_I)
            for [scale] in [scales]
        }
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        if kp is not None:
            raise NotImplementedError
        out_dict = {}
        # import pdb
        # pdb.set_trace()

        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            # rescale to [-1, 1], as genertor use tanh then scale back to [0, 1]
            x[key] = x[key] * 2.0 - 1.0
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict
