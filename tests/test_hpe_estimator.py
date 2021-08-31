from modules.head_pose_exp_estimator import HeadPoseExpEstimator
from .unitest_setup import SetupTestCase
import torch


class TestHPE(SetupTestCase):
    def test_hpe(self):
        config = self.config['model_params']['hpe_estimator_params']
        net = HeadPoseExpEstimator(**config)

        data = torch.ones(
            2, 3, int(32 / config['scale_factor']), int(32 / config['scale_factor']))
        out = net(data)
        num_rot_bins = config['num_rot_bins']
        num_kp = config['num_kp']
        n = 2
        self.assertTrue(
            out['rot']['logits'].shape == (n, 3, num_rot_bins)
        )
        self.assertTrue(
            out['rot']['eulers'].shape == (n, 3)
        )
        self.assertTrue(
            out['rot']['mat'].shape == (n, 3, 3)
        )
        self.assertTrue(
            out['trans'].shape == (n, 3)
        )
        self.assertTrue(
            out['exp'].shape == (n, num_kp, 3)
        )
