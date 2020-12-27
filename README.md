# K-FAC: Kronecker-Factored Approximate Curvature

This repository contains a Pytorch implementation of K-FAC, along with certain variants, i.e.:
* One-Level K-FAC (see [`optimizers/kfac.py`](optimizers/kfac.py) and [paper](https://arxiv.org/abs/1503.05671))
* One-Level E-KFAC (see [`optimizers/ekfac.py`](optimizers/ekfac.py) and [paper](https://arxiv.org/abs/1806.03884))
* Two-Level K-FAC (see [`optimizers/gkfac.py`](optimizers/gkfac.py) and [paper](https://arxiv.org/abs/2011.00573))

**Note:** The current implementations support only single-GPU training.

The main scripts provided are the following:
* [`train_cifar.py`](train_cifar.py): Benchmarks optimizers on training different networks on CIFAR10 and CIFAR100 datasets.
* [`train_mnist.py`](train_mnist.py): Benchmarks optimizers on training a deep autoencoder network on MNIST dataset.
* [`train_toy.py`](train_toy.py): Benchmarks optimizers on training linear and non-linear MLPs on Gaussian data with planted targets.

For details on the command-line arguments check the file [`utils/get_args.py`](utils/get_args.py).

To perform grid-search use the script [`grid_search.py`](grid_search.py).

## References

Please consider citing the following papers for K-FAC:
```
@inproceedings{martens2015optimizing,
  title={Optimizing neural networks with kronecker-factored approximate curvature},
  author={Martens, James and Grosse, Roger},
  booktitle={International conference on machine learning},
  pages={2408--2417},
  year={2015}
}

@inproceedings{grosse2016kronecker,
  title={A kronecker-factored approximate fisher matrix for convolution layers},
  author={Grosse, Roger and Martens, James},
  booktitle={International Conference on Machine Learning},
  pages={573--582},
  year={2016}
}
```

and for E-KFAC:

```
@inproceedings{george2018fast,
  title={Fast Approximate Natural Gradient Descent in a Kronecker Factored Eigenbasis},
  author={George, Thomas and Laurent, C{\'e}sar and Bouthillier, Xavier and Ballas, Nicolas and Vincent, Pascal},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9550--9560},
  year={2018}
}
```

For two-level K-FAC please cite:
```
@misc{tselepidis2020twolevel,
  title={Two-Level K-FAC Preconditioning for Deep Learning},
  author={Nikolaos Tselepidis and Jonas Kohler and Antonio Orvieto},
  year={2020},
  eprint={2011.00573},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Contact

If you have any questions or suggestions, please feel free to contact me via email at ntselepidis@student.ethz.ch.
