from torch import Tensor, optim
from runlib.logger import Logger
from tqdm import trange
import torch

from torch.utils.data import DataLoader
from modules import GeneratorFullModel, DiscriminatorFullModel, head_pose_exp_estimator
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from runlib.frames_dataset import DatasetRepeater
from modules import (
    ApearanceEncoder,
    HeadPoseExpEstimator,
    CanonicalKPDetector,
    OcclusionEstimator,
    OcclAwareGenerator,
    MultiScaleDiscriminator,
    GeneratorFullModel,
    DiscriminatorFullModel,
)
from .frames_dataset import FramesDataset
from .__init__ import NAME_LIST
import torchvision


def train(config, appearance_encoder: ApearanceEncoder, hpe_estimator: HeadPoseExpEstimator, kp_detector: CanonicalKPDetector, occlusion_estimator: OcclusionEstimator, generator: OcclAwareGenerator, discriminator: MultiScaleDiscriminator, checkpoint, log_dir: str, dataset: FramesDataset, num_workers: int, use_cuda: bool):

    train_params = config['train_params']

    model_list = [
        appearance_encoder,
        hpe_estimator,
        kp_detector,
        occlusion_estimator,
        generator,
        discriminator
    ]

    # optimizer
    optimizer_appearance_encoder = torch.optim.Adam(
        appearance_encoder.parameters(),
        lr=train_params['lr_appearance_encoder'],
        betas=(0.5, 0.999)
    )
    optimizer_hpe_estimator = torch.optim.Adam(
        hpe_estimator.parameters(),
        lr=train_params['lr_hpe_estimator'],
        betas=(0.5, 0.999)
    )
    optimizer_kp_detector = torch.optim.Adam(
        appearance_encoder.parameters(),
        lr=train_params['lr_kp_detector'],
        betas=(0.5, 0.999)
    )
    optimizer_occlusion_estimator = torch.optim.Adam(
        appearance_encoder.parameters(),
        lr=train_params['lr_occlusion_estimator'],
        betas=(0.5, 0.999)
    )
    optimizer_generator = torch.optim.Adam(
        appearance_encoder.parameters(),
        lr=train_params['lr_generator'],
        betas=(0.5, 0.999)
    )
    optimizer_discriminator = torch.optim.Adam(
        appearance_encoder.parameters(),
        lr=train_params['lr_discriminator'],
        betas=(0.5, 0.999)
    )
    optimizer_list = [
        optimizer_appearance_encoder,
        optimizer_hpe_estimator,
        optimizer_kp_detector,
        optimizer_occlusion_estimator,
        optimizer_generator,
        optimizer_discriminator
    ]
    print("[Finished init Optimizers]")

    # load checkpoint
    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            *model_list, *optimizer_list
        )
        print("[Finished load weights]")
    else:
        start_epoch = 0
        print("[No need to load weights]")

    # optimizer scheduler
    scheduler_appearance_encoder = MultiStepLR(
        optimizer_appearance_encoder, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_hpe_estimator = MultiStepLR(
        optimizer_hpe_estimator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(
        optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1, last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_occlusion_estimator = MultiStepLR(
        optimizer_occlusion_estimator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_generator = MultiStepLR(
        optimizer_generator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(
        optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_list = [
        scheduler_appearance_encoder,
        scheduler_hpe_estimator,
        scheduler_kp_detector,
        scheduler_occlusion_estimator,
        scheduler_generator,
        scheduler_discriminator
    ]
    print("[Finished init scheduler]")

    # hopenet, psudo-head-rot-label
    from modules.hopenet.hopenet import Hopenet
    from modules.hopenet.utils import softmax_temperature
    hopenet = Hopenet(torchvision.models.resnet.Bottleneck,
                      [3, 4, 6, 3], 66)
    hopenet_state_dict = torch.load(train_params['hopenet_weight_path'])
    hopenet.load_state_dict(hopenet_state_dict)
    hopenet = hopenet.cuda()
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).view(1, -1, 1).cuda()
    print("[Finished loading hopenet]")

    # dataloader
    if 'num_repeats' in train_params or train['num_repeats'] > 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(
        dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=num_workers, drop_last=True)
    print("[Finished creating dataloader]")

    # train full path for gen & disc
    generator_full = GeneratorFullModel(
        appearance_encoder, kp_detector, hpe_estimator, occlusion_estimator, generator, discriminator, train_params)
    if use_cuda and sum(generator_full.loss_weights['perceptual']) != 0:
        generator_full.vgg = generator_full.vgg.cuda()
    # if use_cuda and sum(generator_full.loss_weights['face_perceptual']) != 0:
    #     generator_full.face_vgg = generator_full.face_vgg.cuda()
    discriminator_full = DiscriminatorFullModel(
        appearance_encoder, kp_detector, hpe_estimator, occlusion_estimator, generator, discriminator, train_params)
    generator_full = generator_full.cuda()
    discriminator_full = discriminator_full.cuda()
    if use_cuda:
        generator_full = DataParallelWithCallback(
            generator_full, device_ids=list(range(torch.cuda.device_count())))
        discriminator_full = DataParallelWithCallback(
            discriminator_full, device_ids=list(range(torch.cuda.device_count())))
        print("[Finished moving to GPUs]")

    print("[Start training]")
    # train
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                x = {k: v.cuda()
                     for k, v in x.items() if isinstance(v, Tensor)}

                # y ...
                with torch.no_grad():
                    y = {}
                    for key in ('source', 'driving'):
                        y[key] = torch.stack(hopenet(x[key]), -1)
                        y[key] = softmax_temperature(y[key], 1)
                        y[key] = torch.sum(y[key] * idx_tensor, dim=1) * 3 - 99

                loss_dict, out_dict = generator_full(x, y)
                loss_values = [
                    val.mean() for val in loss_dict.values()
                ]
                loss: Tensor = sum(loss_values)

                # update gen part in GAN
                for model, optimizer in zip(model_list, optimizer_list):
                    if not isinstance(model, MultiScaleDiscriminator):
                        optimizer.zero_grad()
                loss.backward()
                for model, optimizer in zip(model_list, optimizer_list):
                    if not isinstance(model, MultiScaleDiscriminator):
                        optimizer.step()

                # update disc part in GAN
                if train_params['loss_weights']['gan_loss_weight'] > 0:
                    loss_disc_dict = discriminator_full(x, out_dict['rgb'])
                    loss_values: Tensor = [
                        val.mean() for val in loss_disc_dict.values()
                    ]
                    loss = sum(loss_values)

                    optimizer_discriminator.zero_grad()
                    loss.backward()
                    optimizer_discriminator.step()

                    loss_dict.update(loss_disc_dict)

                # log iter
                losses_scalar = {
                    key: value.mean().detach().data.cpu().numpy()
                    for key, value in loss_dict.items()
                }
                logger.log_iter(losses=losses_scalar)

            for scheduler in scheduler_list:
                scheduler.step()

            logger.log_epoch(
                epoch,
                {**{k: v for k, v in zip(NAME_LIST, model_list)}, **{
                    f"optimizer_{k}": v for k, v in zip(NAME_LIST, optimizer_list)}},
                inp=x,
                out=out_dict
            )
