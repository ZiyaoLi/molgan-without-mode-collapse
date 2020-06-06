# MolGAN without Mode Collapse
Resolving Mode Collapse in [MolGAN](https://github.com/nicola-decao/MolGAN/tree/master), a Tensorflow implementation. 
See MolGAN in [reference](https://arxiv.org/abs/1805.11973).

## Overview

MolGAN is a Generative Adversarial Network on graph-represented molecular data. See the paper for more information: 
[MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973)

In this repo we resolved the mode collapse reported in the original paper.
Specifically, we established a *conditional trainer* to feed data with higher rewards in the second training phase of MolGAN.

We also implemented other batch-discriminator tricks and a (Relational) Graph Attention Network discriminator.

## Dependencies

* **python>=3.6**
* **tensorflow>=1.7.0**: https://tensorflow.org
* **rdkit**: https://www.rdkit.org
* **numpy**
* **scikit-learn**
* **matplotlib**

## Structure
* [`data`](https://github.com/ZiyaoLi/molgan-without-mode-collapse/tree/master/data): 
should contain your datasets. 
    
    Run `download_dataset.sh` to download QM9 dataset (used for the paper). 
    
    Run `generate_dataset.py` to generate the required format of data after download. 
    (Data in `.sdf` or `.smi` format from other resources can also be transformed using this script.)

    Change the `size` argument in `data.generate` for subset of data.

    `SparseMolecularDataset` generates required data in official MolGAN, 

    `SparseMolecularDatasetWithRewards` generates data required in our conditional training scheme.

* [`molgan_train.py`](https://github.com/ZiyaoLi/molgan-without-mode-collapse/tree/master/molgan_train.py): Original training scheme in MolGAN.
* [`conditional_train.py`](https://github.com/ZiyaoLi/molgan-without-mode-collapse/tree/master/conditional_train.py): Conditional training scheme proposed in this repo.
* [`models`](https://github.com/ZiyaoLi/molgan-without-mode-collapse/tree/master/models): Class for Models. Both VAE and (W)GAN are implemented.
* [`optimizers`](https://github.com/ZiyaoLi/molgan-without-mode-collapse/tree/master/optimizers): Class for Optimizers for both VAE, (W)GAN and RL.
* [`utils`](https://github.com/ZiyaoLi/molgan-without-mode-collapse/tree/master/utils): Utils including model layers, datasets, molecular metrics, reward functions, et al.

## Usage
Please refer to the example [`train.sh`](https://github.com/ZiyaoLi/molgan-without-mode-collapse/tree/master/train.sh).

Use `python molgan_train.py --help` or `python conditional_train.py --help` to check the arguments.

## Notes
VAE models are directly forked from MolGAN and left untested.

