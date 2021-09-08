import pdb
from numpy.lib.utils import source
from torch import nn
import torch
import torch.nn.functional as F
from modules.utils import ImagePyramide, Transform, get_face_keypoint, get_multi_sample_grid, make_coordinate_grid, warp_multi_feature_volume
from torchvision import models
import numpy as np
from torch.autograd import grad
from modules.appearance_encoder import ApearanceEncoder
from modules.head_pose_exp_estimator import HeadPoseExpEstimator
from modules.kp_detector import CanonicalKPDetector
from modules.generator import OcclAwareGenerator
from modules.occlusion_estimator import OcclusionEstimator
from modules.discriminator import MultiScaleDiscriminator
from typing import Dict


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
        inputs:
                x: (n, c, h, w)
        outputs:
                out features: [
                        (n, c_i, h_i, w_i) for [i] in [depths]
                ]
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        print("[WARNING] not use imagenet pre-train weights of VGG19")
        vgg_pretrained_features = models.vgg19(pretrained=False).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class GeneratorFullModel(nn.Module):
    """
    generator full pass
    inputs:
        x: (n, c, h, w)
    outpus:
        disc_maps: (n, 1, h_d, w_d)

    losses:
        perceptual loss (10)
            - each loss weights are (0.03125, 0.0625, 0.125, 0.25, 1.0)
            relu_1_1, _relu_2_1, _relu_3_1, _relu_4_1, _relu_5_1
            - 1 scale face vgg feature. weight?
        gan loss (1)
            1 scale for 256x256
            2 scales for 512x512
            feature matching
        equivariance loss (20)
        keypoint prior loss (10)
        head pose loss (20)
        deformation prior loss (5)

        weights: 10, 1, 20, 10, 20, 5
    """

    def __init__(self, appearance_encoder: ApearanceEncoder, kp_extractor: CanonicalKPDetector, hpe_estimator: HeadPoseExpEstimator, occlusion_estimator: OcclusionEstimator, generator: OcclAwareGenerator, discriminator: MultiScaleDiscriminator, train_params: Dict):
        super(GeneratorFullModel, self).__init__()
        self.appearance_encoder = appearance_encoder
        self.kp_extractor = kp_extractor
        self.hpe_estimator = hpe_estimator
        self.occlusion_estimator = occlusion_estimator
        self.generator = generator
        self.discriminator = discriminator

        self.perceptual_loss_scales = train_params['perceptual_loss_scales']
        self.disc_scales = self.discriminator.scales

        assert(self.perceptual_loss_scales <= self.disc_scales)

        self.pyramid = ImagePyramide(
            self.disc_scales, appearance_encoder.in_features)
        # if torch.cuda.is_available():
        #     self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.train_params = train_params

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()

        if self.loss_weights['face_perceptual'] != 0:
            raise NotImplementedError
            self.face_vgg = Vgg19()

    def forward(self, x):
        """
        generator full pass
        inputs:
            dict: {
                "source": (1, c, h, w)
                "driving": (1, c, h, w)
            }
        outpus:
            disc_maps: (n, 1, h_d, w_d)

        losses:
            perceptual loss (10)
                - each loss weights are (0.03125, 0.0625, 0.125, 0.25, 1.0)
                relu_1_1, _relu_2_1, _relu_3_1, _relu_4_1, _relu_5_1
                - 1 scale face vgg feature. weight?
            gan loss (1)
                1 scale for 256x256
                2 scales for 512x512
            equivariance loss (20)
            keypoint prior loss (10)
            head pose loss (20)
            deformation prior loss (5)

            weights: 10, 1, 20, 10, 20, 5
        """
        can_kp_source_dict = self.kp_extractor(x['source'])
        can_kp_driving_dict = self.kp_extractor(x['driving'])
        """        
        - out: dict{
            "keypoint": kp: (n, num_kp, 3)
            (optional) "jacobian": jacobian: (n, num_kp, 2, 2) or (n, num_kp, 3, 3)
        }
        """

        hpe_source_dict = self.hpe_estimator(x['source'])
        hpe_driving_dict = self.hpe_estimator(x['driving'])
        """
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

        app_source_features = self.appearance_encoder(x['source'])
        # app_driving_features = self.appearance_encoder(x['driving'])
        """
        f_s: (n, block_expansion, d, h/num_down_blocks, w/num_down_blocks)
        """

        kp_source = get_face_keypoint(
            can_kp_source_dict['keypoint'], hpe_source_dict['rot']['mat'], hpe_source_dict['trans'], hpe_source_dict['exp'])
        kp_driving = get_face_keypoint(
            can_kp_driving_dict['keypoint'], hpe_driving_dict['rot']['mat'], hpe_driving_dict['trans'], hpe_driving_dict['exp'])

        # [NOTE] all warping steps are finished in `self.occlusion_estimator`
        # grids = get_multi_sample_grid(app_source_features, source_keypoint=kp_source, target_keypoint=kp_driving,
        #                               source_rot=hpe_source_dict['rot']['mat'], target_rot=hpe_driving_dict['rot']['mat'])
        # warped_k_features = warp_multi_feature_volume(
        #     app_source_features, grids)

        occ_field_dict = self.occlusion_estimator(
            app_source_features, kp_source, kp_driving, hpe_source_dict['rot']['mat'], hpe_driving_dict['rot']['mat'])
        """
        flow occlusion estimator
        input:
            features (n, c, d, h, w)
            source_keypoint (n, num_kp, 3)
            target_keypoint (n, num_kp, 3)
            source_rot (n, 3, 3)
            target_rot (n, 3, 3)
        intermediate network input:
            (num_kp + 1) grid (n, num_kp + 1, d, h, w, 3)
            (num_kp + 1) warped-features, 1 means no warping (n, num_kp + 1, c, d, h, w)
        output:
            3d flow mask: (n, num_kp + 1, d, h, w), softmax
            2d occlusion mask: (n, 1, h , w), sigmoid
        """

        fake = self.generator(app_source_features, kp_source, kp_driving,
                              hpe_source_dict['rot']['mat'], hpe_driving_dict['rot']['mat'], **occ_field_dict)
        """
        Occlusion-aware generator
        inputs:
            features: (n, c, d, h, w)
            source_keypoint: (n, num_kp, 3)
            target_keypoint: (n, num_kp, 3)
            source_rot: (n, 3, 3)
            target_rot: (n, 3, 3)
            flow_3d_mask: (n, num_kp + 1, d, h, w), softmax
            feature_2d_mask: (n, 1, h , w), sigmoid
        outputs:
            rgb: (n, 3, h, w) re-scale to (0, 1) after tanh (-1, 1)
        """

        pyramide_real = self.pyramid(x['driving'])
        pyramide_fake = self.pyramid(fake)
        out_dict = {
            'rgb': fake,  # gan, perceptual
            'kp_source': kp_source,  # equivariance loss?
            'kp_driving': kp_driving,  # equivariance loss?
            'rot_source': hpe_source_dict['rot']['eulers'],  # head pose loss
            'rot_driving': hpe_driving_dict['rot']['eulers'],  # head pose loss
            'deform_source': hpe_source_dict['exp'],
            'deform_driving': hpe_driving_dict['exp']
        }
        # loss part
        loss_values_dict = {}

        if sum(self.loss_weights['perceptual']) > 0:
            value_total = 0
            for scale in self.perceptual_loss_scales:
                x_vgg = self.vgg(pyramide_fake['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values_dict['perceptual_loss'] = value_total

        if self.loss_weights['face_perceptual'] > 0:
            raise NotImplementedError

        if self.loss_weights['gan_loss_weight'] > 0:
            disc_maps_fake = self.discriminator(pyramide_fake)
            disc_maps_real = self.discriminator(pyramide_real)

            value_total = 0
            for scale in self.disc_scales:
                key = f"prediction_map_{scale}"
                value = ((1 - disc_maps_fake[key]) ** 2).mean()
                value_total += self.loss_weights['gan_loss_weight'] * value
            loss_values_dict['gan_loss'] = value_total

            if sum(self.loss_weights['feature_matching']) > 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = f"feature_maps_{scale}"

                    for i, (a, b) in enumerate(zip(disc_maps_real[key], disc_maps_fake[key])):
                        if self.loss_weights['feature_matching'][i] > 0:
                            value = torch.abs(a - b).mean()
                            value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values_dict['feature_matching_loss'] = value_total

        if self.loss_weights['equivariance_loss_weight'] > 0:
            transform = Transform(x['driving'].shape[0],
                                  **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_2d_kp = self.kp_extractor(transformed_frame)[
                'keypoint'][..., :2]

            out_dict['transformed_frame'] = transformed_frame
            out_dict['transformed_kp'] = transformed_2d_kp

            if self.loss_weights['equivariance_loss_weight'] > 0:
                value = torch.abs(
                    kp_driving[..., :2] - transform.warp_coordinates(transformed_2d_kp)).mean()
                loss_values_dict['equivariance_loss'] = self.loss_weights['equivariance_loss_weight'] * value

            # [NOTE] what about jacobian?
        return loss_values_dict, out_dict


class DiscriminatorFullModel(nn.Module):
    def __init__(self, appearance_encoder, kp_extractor, hpe_estimator, occlusion_estimator, generator, train_params):
        super(DiscriminatorFullModel, self).__init__()
