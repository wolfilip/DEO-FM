# Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation (CVPR 2026)

PyTorch implementation and pretrained models for the method DEO, from the paper Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation.

<div align="center">
  <image src="assets/splash_image.png" width="600px" />
  <p></p>
</div>

Authors: [Filip Wolf](https://www.vicos.si/people/#), [Blaž Rolih](https://www.vicos.si/people/blaz_rolih/), [Luka Čehovin Zajc.](https://www.vicos.si/people/luka_cehovin_zajc/)

[[`Paper`](https://arxiv.org/abs/2602.19863)] [[`Project Page`](https://wolfilip.github.io/DEO/)] [[`BibTeX`](#citing-deo)]

## Update
[3-2-2026] We release pretrained weights for DEO.
[26-2-2026] We release pretraining code for DEO.

## Pretrained models

The Swin-b model used for the results in the paper can be found [`here`](https://unilj-my.sharepoint.com/:f:/g/personal/filip_wolf_fri1_uni-lj_si/IgCsJwuKs8-PQIp706uUiPpsAVkaNsVH5pPLAU_6mYkF22Q?e=nUocgl). ViT-based and larger models are coming soon.

## Installation

Our implementation requires Python 3.11+, PyTorch 2.4+ and [xFormers](https://github.com/facebookresearch/xformers) 0.0.29+ and some other packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup the dependencies, please install via:

```sh
conda env create -f environment.yml
```

## Data preparation

We use a mix of [fMoW-Sentinel](https://arxiv.org/abs/2207.08051) and [fMoW-RGB](https://arxiv.org/abs/1711.07846). Refer to the paper for more details.

## Training

To train our model, simply run:

```shell
python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --master_port 1234 DEO/main_dino.py --arch swin_t --batch_size 64 --data_split 500000 --local_crops_number 10 --eps 0.05 --output_dir output/[output_dir] --data_path ../pretraining_dataset.h5 --ms_arch --dist_arch dinov3 --dist_arch_size base
```

We copress our dataset into the .h5 format, and our dataloader expects such a class. However, adding your own dataset loader can be easily configured in `utils/datasets.py`

## Evaluation

TODO

## Citing DEO

If you like what we do, please consicer citing us:
```
@article{wolf2026brewing,
  title={Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation},
  author={Wolf, Filip and Rolih, Blaž and Čehovin Zajc, Luka},
  journal={arXiv preprint arXiv:2602.19863},
  year={2026},
  url={https://arxiv.org/abs/2602.19863}
}
```

## Acknowledgements

This project is largely built upon [SimDINO](https://github.com/RobinWu218/SimDINO?tab=readme-ov-file#citing-simdino), which was in turn built on the original [DINO](https://github.com/facebookresearch/dino) and [DINOv2](https://github.com/facebookresearch/dinov2) projects.