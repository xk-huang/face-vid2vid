import pdb
from .unitest_setup import SetupTestCase

import torch
from modules.kp_detector import CanonicalKPDetector
import yaml


class TestCanonicalKPDetector(SetupTestCase):
    def test_canonicalkpdetector(self):
        config = self.config
        ckp_net = CanonicalKPDetector(
            **config['model_params']['kp_detector_params'])

        data = torch.empty(
            2, config['model_params']['kp_detector_params']['num_in_channels'], 64, 64)

        out = ckp_net(data)

        print("[test]", out['keypoint'].shape, (
            data.shape[0], config['model_params']['kp_detector_params']['num_kp'], 3))
        self.assertTrue(out['keypoint'].shape == (
            data.shape[0], config['model_params']['kp_detector_params']['num_kp'], 3))
