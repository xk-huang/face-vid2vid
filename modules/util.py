from numpy.lib.utils import source
from torch.functional import align_tensors
import torchvision.models.resnet
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d, SynchronizedBatchNorm3d as BatchNorm3d
from torch import nn

import torch.nn.functional as F
import torch
from typing import List

print("[WARNING] USE BN with torch.nn.dataparallel, which is quite slow.\nre-implementation required.")


def make_coordinate_grid(spatial_size, dtype):
    """
    input:
        spatial_size: 2d or 3d
        dtype
    output:
        *spatial_size, 3
    """
    spatial_size = [
        None, *spatial_size] if len(spatial_size) == 2 else spatial_size
    print(f"[test] spatial_size {spatial_size}")
    d, h, w = spatial_size
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    if d is not None:
        z = torch.arange(d).type(dtype)
        z = (2 * (z / (d - 1)) - 1)

        xx = x.view(1, 1, -1).repeat(d, h, 1)
        yy = y.view(1, -1, 1).repeat(d, 1, w)
        zz = z.view(-1, 1, 1).repeat(1, h, w)
        grid = [xx.unsqueeze_(-1), yy.unsqueeze_(-1), zz.unsqueeze_(-1)]
    else:
        xx = x.view(1, -1).repeat(h, 1)
        yy = y.view(-1, 1).repeat(1, w)
        grid = [xx.unsqueeze_(-1), yy.unsqueeze_(-1)]

    return torch.cat(grid, -1)


class DownBlock(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_3d=False):
        super(DownBlock, self).__init__()

        conv_func = nn.Conv3d if use_3d else nn.Conv2d
        norm_func = BatchNorm3d if use_3d else BatchNorm2d
        self.conv = conv_func(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = norm_func(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class UpBlock(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_3d=False):
        super(UpBlock, self).__init__()

        conv_func = nn.Conv3d if use_3d else nn.Conv2d
        norm_func = BatchNorm3d if use_3d else BatchNorm2d
        self.conv = conv_func(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = norm_func(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class ResBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size=3, padding=1, use_3d=False):
        super(ResBlock, self).__init__()

        conv_func = nn.Conv3d if use_3d else nn.Conv2d
        norm_func = BatchNorm3d if use_3d else BatchNorm2d

        self.conv1 = conv_func(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = conv_func(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = norm_func(in_features, affine=True)
        self.norm2 = norm_func(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class ResBottleneck(nn.Module):
    """
    ResBottleneck block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, kerner_size=3, padding=1, bottleneck_scale=0.25, use_down_sample=False, use_3d=False):
        super(ResBottleneck, self).__init__()

        self.bottleneck_ch = int(in_features * bottleneck_scale)
        conv_func = nn.Conv3d if use_3d else nn.Conv2d
        norm_func = BatchNorm3d if use_3d else BatchNorm2d

        self.use_down_sample = use_down_sample
        self.conv1 = conv_func(in_channels=in_features,
                               out_channels=self.bottleneck_ch, kernel_size=1)
        self.norm1 = norm_func(self.bottleneck_ch, affine=True)
        self.conv2 = conv_func(in_channels=self.bottleneck_ch, out_channels=self.bottleneck_ch,
                               kernel_size=kernel_size, stride=2 if use_down_sample else 1, padding=padding)
        self.norm2 = norm_func(self.bottleneck_ch, affine=True)
        self.conv3 = conv_func(
            in_channels=self.bottleneck_ch, out_channels=in_features, kernel_size=1)
        self.norm3 = norm_func(in_features, affine=True)

        if use_down_sample:
            self.short_cut = nn.Sequential(
                conv_func(in_channels=in_features,
                          out_channels=in_features, kernel_size=1, stride=2),
                norm_func(in_features, affine=True)
            )
        else:
            self.short_cut = nn.Sequential()

    def foward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)

        if self.use_down_sample:
            x = self.short_cut(x)

        return out + x


class UNetEncoder(nn.Module):
    def __init__(self, use_3d=False, block_expansion=32, in_features=3, num_blocks=5, max_features=1024):
        super(UNetEncoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            in_ch = in_features if i == 0 else min(
                max_features, block_expansion * (2 ** i))
            out_ch = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock(in_ch, out_ch, use_3d=use_3d))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
            print("[test] down_block", outs[-1].shape)
        return outs


class UNetDecoder(nn.Module):
    def __init__(self, use_3d=False, block_expansion=32, in_features=3, num_blocks=5, max_features=1024, use_skip=False):
        super(UNetDecoder, self).__init__()

        up_blocks = []
        self.use_3d = use_3d
        self.use_skip = use_skip

        for i in range(num_blocks)[::-1]:

            in_ch = (1 if i == num_blocks - 1 or not use_skip else 2) * \
                min(max_features, block_expansion * (2 ** (i + 1)))
            out_ch = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock(in_ch, out_ch, use_3d=use_3d))
        self.up_blocks = nn.ModuleList(up_blocks)

        if use_skip:
            self.num_out_ch = (block_expansion + in_features)
        else:
            self.num_out_ch = block_expansion

    def forward(self, x):
        if not isinstance(x, List):
            x = [x]

        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            if self.use_skip:
                skip = x.pop()
                out = torch.cat([out, skip], dim=1)
            print("[test] up_block", out.shape)
        return out


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        # sigma = 1.5
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


def warp_multi_feature_volume(features, grid):
    '''
    input:
        features (n, c, d, h, w)
        grid (n, num_kp + 1, d, h, w, 3)
    output:
        warped features (n, num_kp + 1, c, d, h, w)
    '''
    _, _, d, h, w = features.shape
    _, _, _d, _h, _w, _ = grid.shape
    if d != _d or h != _h or w != _w:
        grid = F.interpolate(
            grid, (d, h, w), mode='trilinear', align_corners=True)
    warped_features = []
    for i in range(grid.shape[1]):
        warped_features.append(F.grid_sample(
            features, grid[:, i], align_corners=True))
    return torch.stack(warped_features).transpose(0, 1)


def get_multi_sample_grid(features, source_keypoint, target_keypoint, source_rot, target_rot):
    """
    flow occlusion estimator
    input:
        features (n, c, d, h, w)
        source_keypoint (n, num_kp, 3)
        target_keypoint (n, num_kp, 3)
        source_rot (n, 3, 3)
        target_rot (n, 3, 3)
    output:
        grid: (n, num_kp, d, h, w, 3)
    """
    n, num_kp, _ = source_keypoint.shape
    d, h, w = features.shape[-3:]
    identity_grid = make_coordinate_grid(
        features.shape[-3:], features.type())  # d, h, w, 3
    identity_grid = identity_grid.view(1, 1, *identity_grid.shape)
    source_keypoint = source_keypoint.view(n, num_kp, 1, 1, 1, 3)
    target_keypoint = target_keypoint.view(n, num_kp, 1, 1, 1, 3)
    source_rot = source_rot.view(
        n, 1, 1, 1, 1, 3, 3).expand(n, num_kp, d, h, w, 3, 3)
    target_rot = target_rot.view(
        n, 1, 1, 1, 1, 3, 3).expand(n, num_kp, d, h, w, 3, 3)
    grid = identity_grid - target_keypoint

    grid = torch.matmul(torch.inverse(target_rot), grid.unsqueeze(-1))
    grid = torch.matmul(source_rot, grid)  # n, num_kp, d, h, w, 3
    grid = grid.squeeze(-1) + source_keypoint

    return torch.cat([grid, identity_grid.expand(n, 1, d, h, w, 3)], dim=1)


def get_face_keypoint(kp, rot, trans, exp):
    """
    input:
        kp: (n, num_kp, 3)
        rot (n, 3, 3)
        trans (n, 3)
        exp (n, num_kp, 3)
    output:
        facial kp (n, num_kp, 3)
    """
    n, num_kp, _ = kp.shape
    out = kp + trans.view(n, 1, -1)
    out = torch.matmul(rot.unsqueeze(1).expand(
        n, num_kp, 3, 3), out.unsqueeze(-1))  # n, num_kp, 3
    out = out.squeeze(-1) + exp
    return out
