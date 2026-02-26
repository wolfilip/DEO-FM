# Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation (CVPR 2026)

PyTorch implementation and pretrained models for DEO.

<div align="center">
  <image src="assets/splash_image.png" width="600px" />
  <p></p>
</div>

Authors: [Filip Wolf](https://www.vicos.si/people/#), [Blaž Rolih](https://www.vicos.si/people/blaz_rolih/), [Luka Čehovin Zajc.](https://www.vicos.si/people/luka_cehovin_zajc/)

[[`Paper`](https://arxiv.org/abs/2602.19863)] [[`Project Page`](https://wolfilip.github.io/DEO/)] [[`BibTeX`](#citing-deo)]

## Update
[26-2-2026] We released pretraining code for DEO.

## Pretrained models
Stay tuned.

## Installation

Our implementation requires Python 3.11+, PyTorch 2.4+ and [xFormers](https://github.com/facebookresearch/xformers) 0.0.29+ and some other packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup the dependencies, please install via:

```sh
conda env create -f environment.yml
```


## Data preparation

TODO


## Training

TODO


## Evaluation

TODO

## Citing DEO

If you find this project useful, please consider giving us a star and citation:
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