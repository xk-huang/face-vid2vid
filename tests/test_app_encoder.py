from typing import Set
import torch
from modules.appearance_encoder import ApearanceEncoder
from .unitest_setup import SetupTestCase


class TestAppEncoder(SetupTestCase):
    def test_appendoer(self):
        app_config = self.config['model_params']['appearance_encoder_params']
        net = ApearanceEncoder(**app_config)

        data = torch.empty(2, 3, 8, 8)
        out = net(data)

        expected_out_shape = (2, app_config['block_expansion'], app_config['depth'], data.shape[-2] //
                              (2 ** app_config['num_down_blocks']), data.shape[-1] // (2 ** app_config['num_down_blocks']))
        print(f"[test] {out.shape} {expected_out_shape}")
        self.assertTrue(out.shape == expected_out_shape)
