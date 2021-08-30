import unittest
import torch
import yaml


class SetupTestCase(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        config_path = 'config/base.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.load(f)
