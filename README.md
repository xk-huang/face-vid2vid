# Face vid2vid

@xiaokehuang@foxmail.com

paper: One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing, cvpr 21

## DEV TODOs

- [ ] Modules

  - [ ] 3d keypoint estimator
    - [ ] facial canonical keypoint estimator
    - [ ] Head pose & exp deformation estimator
          eular2rot
  - [ ] 3d flow estimator
    - [ ] appearance estimator
    - [ ] 3d flow occlusion map + feature occlusion map
  - [ ] generator
    - [ ] appearnce estimator
    - [ ] 3d flow occlusion map + feature occlusion map
    - [ ] feature+occlusion generator
  - [ ] discriminator
    - [ ] multi-scale discriminator
  - [ ] model
    - [ ] 3d keypoint estimator
    - [ ] 3d flow estimator
    - [ ] train_gen_model
    - [ ] train_disc_model
    - [ ] eval_gen_model

- [ ] runlib
  - [ ] train
  - [ ] recontruction
  - [ ] animation
  - [ ] dataset
- [ ] data
  - [ ] a few HD 1024+ voxceleb
  - [ ] Head Pose estimation
  - [ ] full data

## PLAN

- [ ] sample data & full data
- [ ] modules
- [ ] runlib
