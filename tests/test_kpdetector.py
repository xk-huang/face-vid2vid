import pdb
import unittest

import unittest

import torch
from modules.kp_detector import CanonicalKPDetector
import yaml


class TestCanonicalKPDetector(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_canonicalkpdetector(self):
        config_path = 'config/base.yaml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)

        ckp_net = CanonicalKPDetector(
            **config['model_params']['common_params'],
            **config['model_params']['kp_detector_params'])

        data = torch.empty(2, 3, 64, 64)

        out = ckp_net(data)

        print("[test]", out['keypoint'].shape, (
            data.shape[0], config['model_params']['common_params']['num_kp']))
        self.assertTrue(out['keypoint'].shape == (
            data.shape[0], config['model_params']['common_params']['num_kp']))
