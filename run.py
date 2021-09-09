from matplotlib.pyplot import cool
from runlib import train, reconstruction, animate
import torch
from modules import (
    ApearanceEncoder,
    HeadPoseExpEstimator,
    CanonicalKPDetector,
    OcclusionEstimator,
    OcclAwareGenerator,
    MultiScaleDiscriminator,
    GeneratorFullModel,
    DiscriminatorFullModel,
    head_pose_exp_estimator,
)
from runlib.frames_dataset import FramesDataset
from shutil import copy
from argparse import ArgumentParser
from time import gmtime, strftime
import os.path as osp
import yaml
import sys
import os
import matplotlib
matplotlib.use('Agg')


if __name__ == "__main__":
    if sys.version_info[0] < 3:
        raise Exception(
            "You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True,
                        type=str, help="path to config file")
    parser.add_argument("--mode", default="train", type=str,
                        choices=("train", "reconstruction", "animate"))
    parser.add_argument("--log_dir", default="logs", help="path to log")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        help="print verbose info like model arch")
    parser.add_argument("--no_cuda", action="store_true")
    parser.set_defaults(verbose=False)
    parser.set_defaults(no_cuda=False)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        log_dir = osp.join(*osp.split(opt.checkpoint)[:-1])
    else:
        log_dir = osp.join(opt.log_dir, osp.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    # copy config files
    if not osp.path.exists(osp.join(log_dir, os.path.basename)):
        copy(opt.config, log_dir)

    model_config = opt.config['model_params']
    train_config = opt.config['train_params']

    # model settings
    appearance_encoder = ApearanceEncoder(
        **model_config['appearance_encoder_params'])
    hpe_estimator = HeadPoseExpEstimator(
        **model_config['hpe_estimator_params'])
    kp_detector = CanonicalKPDetector(**model_config['kp_detector_params'])
    occlusion_estimator = OcclusionEstimator(
        **model_config['occl_estimator_params'])
    generator = OcclAwareGenerator(**model_config['generator_params'])
    discriminator = MultiScaleDiscriminator(
        **model_config['disciminator_params'])

    if opt.cuda:
        print("[Use CUDA]")
        appearance_encoder = appearance_encoder.cuda()
        hpe_estimator = hpe_estimator.cuda()
        kp_detector = kp_detector.cuda()
        occlusion_estimator = occlusion_estimator.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    else:
        print("[Running on CPU, are you sure about this?")

    if opt.verbose:
        print("[Model Architecture]")
        print(appearance_encoder)
        print(hpe_estimator)
        print(kp_detector)
        print(occlusion_estimator)
        print(generator)
        print(discriminator)

    dataset = FramesDataset(
        is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        print("Training...")
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
    elif opt.mode == 'animate':
        print("Animate...")
    else:
        raise ValueError(f"Wrong `mode` arg: {opt.mode}")
