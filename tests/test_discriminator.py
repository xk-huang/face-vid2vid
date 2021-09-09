import pdb
import torch
from modules.appearance_encoder import ApearanceEncoder
from .unitest_setup import SetupTestCase
from modules.utils import (get_face_keypoint, get_multi_sample_grid,
                           warp_multi_feature_volume, make_coordinate_grid, ImagePyramide)
import torch.nn.functional as F
from modules.occlusion_estimator import OcclusionEstimator
from modules.generator import OcclAwareGenerator
from modules.discriminator import MultiScaleDiscriminator


class TestDisc(SetupTestCase):
    def test_disc(self):
        config = self.config['model_params']
        n, num_kp = 2, config['generator_params']['num_kp']

        occ_net = OcclusionEstimator(**config['occl_estimator_params'])
        gen_net = OcclAwareGenerator(**config['generator_params'])
        disc_net = MultiScaleDiscriminator(**config["disciminator_params"])
        imgpy = ImagePyramide(
            config["disciminator_params"]['scales'], config["disciminator_params"]['num_channels'])

        kp = torch.randn(n, num_kp, 3)
        rot = torch.randn(n, 3, 3)
        trans = torch.randn(n, 3, 1)
        exp = torch.randn(n, num_kp, 3)
        features = torch.randn(
            n, config['occl_estimator_params']['num_features_ch'], config['occl_estimator_params']['depth'], 8, 8)
        face_kp = get_face_keypoint(kp, rot, trans, exp)
        grids = get_multi_sample_grid(features, face_kp, face_kp, rot, rot)
        volumes = warp_multi_feature_volume(features, grids)

        out = occ_net(features, face_kp, face_kp, rot, rot)

        self.assertTrue(torch.allclose(out['flow_3d_mask'].sum(
            dim=(-1, -2, -3)), torch.ones(1)))
        self.assertTrue(torch.all(out['feature_2d_mask'] < 1.0))
        self.assertTrue(torch.all(0.0 < out['feature_2d_mask']))

        rgb = gen_net(features, face_kp, face_kp, rot, rot, **out)
        self.assertTrue(torch.all(rgb < 1.0))
        self.assertTrue(torch.all(0.0 < rgb))

        rgbs = imgpy(rgb)
        disc_out = disc_net(rgbs)

        # import pdb
        # pdb.set_trace()
        # trues = [torch.allclose(features, volumes[:, i], atol=1e-4)
        #          for i in range(num_kp + 1)]
        # # print('[test]', trues)
        # self.assertTrue(
        #     all(trues)
        # )
