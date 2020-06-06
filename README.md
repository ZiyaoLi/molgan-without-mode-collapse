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
* [data](https://github.com/nicola-decao/MolGAN/tree/master/data): 
should contain your datasets. 
    
    Run `download_dataset.sh` to download QM9 dataset (used for the paper). 
If you wish to use this data, run `generate_dataset.py` to generate the required format of data. 
(Change the `size` argument in `data.generate` for subset of data.) 

    `SparseMolecularDataset` generates required data in official MolGAN, 
whereas `SparseMolecularDatasetWithRewards` generates data required in our conditional training scheme.
* [molgan_train](https://github.com/nicola-decao/MolGAN/blob/master/molgan_train.py): Original training scheme in MolGAN.
* [conditional_train](https://github.com/nicola-decao/MolGAN/blob/master/conditional_train.py): Conditional training scheme proposed in this repo.
* [models](https://github.com/nicola-decao/MolGAN/tree/master/models): Class for Models. Both VAE and (W)GAN are implemented.
* [optimizers](https://github.com/nicola-decao/MolGAN/tree/master/optimizers): Class for Optimizers for both VAE, (W)GAN and RL.

## Usage
Please have a look at the [example](https://github.com/nicola-decao/MolGAN/blob/master/example.py).

Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Nicola De Cao](mailto:nicola.decao@gmail.com).

## License
MIT

## Citation
```
[1] De Cao, N., and Kipf, T. (2018).MolGAN: An implicit generative 
model for small molecular graphs. ICML 2018 workshop on Theoretical
Foundations and Applications of Deep Generative Models.
```

BibTeX format:
```
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small
  molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations 
  and Applications of Deep Generative Models},
  year={2018}
}

```
