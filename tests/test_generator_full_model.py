import pdb
import torch
from modules.appearance_encoder import ApearanceEncoder
from .unitest_setup import SetupTestCase
from modules.utils import get_face_keypoint, get_multi_sample_grid, warp_multi_feature_volume, make_coordinate_grid
import torch.nn.functional as F
from modules.occlusion_estimator import OcclusionEstimator
from modules.generator import OcclAwareGenerator
from modules.appearance_encoder import ApearanceEncoder
from modules.head_pose_exp_estimator import HeadPoseExpEstimator
from modules.kp_detector import CanonicalKPDetector
from modules.generator import OcclAwareGenerator
from modules.occlusion_estimator import OcclusionEstimator
from modules.discriminator import MultiScaleDiscriminator
from modules.model import GeneratorFullModel
from typing import Dict


class TestGenFull(SetupTestCase):
    def test_gen_full(self):
        model_config = self.config['model_params']
        train_config = self.config['train_params']
        app_net = ApearanceEncoder(**model_config['appearance_encoder_params'])
        hpe_net = HeadPoseExpEstimator(**model_config['hpe_estimator_params'])
        kp_net = CanonicalKPDetector(**model_config['kp_detector_params'])
        occ_net = OcclusionEstimator(**model_config['occl_estimator_params'])
        gen_net = OcclAwareGenerator(**model_config['generator_params'])
        disc_net = MultiScaleDiscriminator(
            **model_config['disciminator_params'])

        gen_full_model = GeneratorFullModel(
            app_net, kp_net, hpe_net, occ_net, gen_net, disc_net, train_config)

        h, w = 64, 64
        data = {
            'source': torch.empty(1, 3, h, w),
            'driving': torch.empty(1, 3, h, w)
        }
        loss_dict, out_dict = gen_full_model(data)
        import pdb
        pdb.set_trace()
