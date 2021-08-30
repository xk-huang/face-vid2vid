import pdb
from typing import Set
import torch
from modules.appearance_encoder import ApearanceEncoder
from .unitest_setup import SetupTestCase
from modules.util import get_face_keypoint, get_multi_sample_grid, warp_multi_feature_volume, make_coordinate_grid
import torch.nn.functional as F


class TestAppEncoder(SetupTestCase):
    def test_appendoer(self):
        n, num_kp = 2, 5

        kp = torch.randn(n, num_kp, 3)
        rot = torch.randn(n, 3, 3)
        trans = torch.randn(n, 3, 1)
        exp = torch.randn(n, num_kp, 3)
        features = torch.randn(n, 3, 6, 6, 6)
        face_kp = get_face_keypoint(kp, rot, trans, exp)
        grids = get_multi_sample_grid(features, kp, kp, rot, rot)
        volumes = warp_multi_feature_volume(features, grids)

        # torch.allclose(, )
        # import pdb
        # pdb.set_trace()
        trues = [torch.allclose(features, volumes[:, i], atol=1e-4)
                 for i in range(num_kp + 1)]
        print('[test]', trues)
        self.assertTrue(
            all(trues)
        )
