import torch
import torch.nn as nn
from modules.utils import ResBottleneck, AntiAliasInterpolation2d
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d, SynchronizedBatchNorm3d as BatchNorm3d
from torchvision.models import resnet50
import numpy as np


class HeadPoseExpEstimator(nn.Module):
    """head pose & expression
    input
        x: (n, c, h, w)

    output:
        dict{
            'rot': {
                'logits': logits (n, 3, num_rot_bins)  # yaw pitch roll
                'eulers': eulers (n, 3)  # yaw pitch roll
                'mat': mat (n, 3, 3)
            },
            'trans': trans (n, 3)
            'exp': exp (n, num_kp, 3)
        }
    """

    def __init__(self, pretrained=False, num_kp=5, num_rot_bins=66, half_range=99, scale_factor=1):
        super(HeadPoseExpEstimator, self).__init__()

        self.register_buffer('idx_tensor', torch.arange(num_rot_bins).float())
        self.angle_names = ('yaw', 'pitch', 'roll')
        self.num_rot_bins = num_rot_bins
        self.half_range = half_range
        self.bin_size = half_range // num_rot_bins

        self.net = resnet50(pretrained=pretrained)
        in_ch = self.net.fc.weight.shape[-1]

        # rot_linear = []
        # for i in range(3):
        #     rot_linear.append(nn.Linear(in_ch, num_rot_bins))
        # self.rot_layers = nn.ModuleList(rot_linear)
        self.euler_bin = nn.Linear(in_ch, 3 * num_rot_bins)

        self.trans = nn.Linear(in_ch, 3)
        self.exp = nn.Linear(in_ch, num_kp * 3)

        self.scale_factor = scale_factor
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(
                self.net.conv1.weight.shape[1], scale_factor)

    # def get_values_dict(self, rot_logits):
    #     return {
    #         k: torch.sum(v * self.idx_tensor, -1) *
    #         self.bin_size - self.half_range
    #         for k, v in rot_logits.items()
    #     }
    def get_eulers(self, rot_logits):
        return torch.sum(rot_logits * self.idx_tensor, -1) * self.bin_size - self.half_range

    def get_rot_mat(self, rot_eulers):
        # inputs: yaw - Y, pitch - X, roll - Z
        # [WARNING] according to [Ruiz.18] Showhow X & Y should be times -1, empirically verified
        rot_eulers = rot_eulers[:, (1, 0, 2)]
        rot_eulers = rot_eulers * np.pi / 180.0
        for i in range(2):
            rot_eulers[:, i] *= -1.0
        return self.euler2rot(rot_eulers)

    def euler2rot(self, euler_angle):
        # inputs: pitch - X, yaw - Y, roll - Z
        batch_size = euler_angle.shape[0]
        theta = euler_angle[:, 0].reshape(-1, 1, 1)
        phi = euler_angle[:, 1].reshape(-1, 1, 1)
        psi = euler_angle[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                         device=euler_angle.device)
        zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                           device=euler_angle.device)
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ), 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
        ), 2)
        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)

        # rot_logits = {}
        # for i, angle_name in enumerate(self.angle_names):
        #     rot_logits[angle_name] = self.rot_layers[i](x)
        # rot_values = self.get_values_dict(rot_logits)
        # rot_mat = self.get_rot_mat(rot_values)
        rot_logtis = self.euler_bin(x)
        rot_logtis = rot_logtis.view(-1, 3, self.num_rot_bins)
        rot_eulers = self.get_eulers(rot_logtis)
        rot_mat = self.get_rot_mat(rot_eulers)

        trans = self.trans(x)
        exp = self.exp(x)

        return {
            'rot': {
                'logits': rot_logtis,
                'eulers': rot_eulers,
                'mat': rot_mat
            },
            'trans': trans,
            'exp': exp.view(exp.shape[0], -1, 3)
        }
