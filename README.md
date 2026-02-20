# Complexity Analysis of Normalizing Constant Estimation: from Jarzynski Equality to Annealed Importance Sampling and beyond ([ICLR 2026](https://openreview.net/forum?id=96fJALwotm))

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2502.04575)
[![X](https://img.shields.io/badge/X-000000?logo=x&logoColor=white&style=flat-square)](https://x.com/WeiGuo01/status/1889721841142661538?s=20)

This repository contains the code for the experiment in our paper *Complexity Analysis of Normalizing Constant Estimation: from Jarzynski Equality to Annealed Importance Sampling and beyond*, accepted at ICLR 2026.

## Setting up the environment

To install the required packages, run the following command:

```bash
pip install ConfigArgParse matplotlib numpy PyYAML torch tqdm wandb Cython POT pytorch-minimize torchquad
```

## Replicating our results

You may replicate all our results by running the following commands:

```bash
python run.py --alg <algorithm_name> --dist <distribution_name>
```
where `<algorithm_name>` is chosen from `'ti', 'ais', 'rdmc', 'rsdmc', 'zodmc', 'sndmc'`, and `<distribution_name>` is chosen from `'2d_gmm', 'mueller'`.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{guo2026complexity,
  title     = {Complexity Analysis of Normalizing Constant Estimation: from {Jarzynski} Equality to Annealed Importance Sampling and beyond},
  author    = {Wei Guo and Molei Tao and Yongxin Chen},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=96fJALwotm}
}
```

## Acknowledgements

This repo is partially based on the [code](https://github.com/KevinRojas1499/ZOD-MC/) of the paper [Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion](https://openreview.net/forum?id=X3Aljulsw5).
