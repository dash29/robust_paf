# robust_paf
Code for Paper Parameterizing Activation Functions for Adversarial Robustness

In addition to CIFAR-10 and CIFAR-100, the datasets used in this paper are publicly available:
- [DDPM-6M](https://github.com/preetum/cifar5m)
- [TI-500K](https://github.com/yaircarmon/semisup-adv)
- [ImageNette](https://github.com/fastai/imagenette)

Standard configuration files are included in configs directory.  Code assumes datasets are saved in a subdirectory named data.  To change the path, you can modify the config file.  For DDPM-6M and TI-500K, the path must be modified in utils.py get_synthetic_dataloader and tinyimages500k_dataset functions.

For standard training experiments on CIFAR-10:
```
python train.py --configs configs/configs_cifar10.yml --exp-name EXPERIMENT_NAME --fix-act --fix-act-val VALUE_TO_SET_PAF_ALPHA --trainer baseline --val-method baseline --arch resnet18 --activation ACTIVATION
```
For evaluation of standard trained models:
```
python eval_std.py --configs configs/configs_cifar10.yml --exp-name EXPERIMENT_NAME --arch resnet18 --activation ACTIVATION --ckpt PATH_TO_CHECKPOINT_FILE
```


To train with PGD adversarial training on CIFAR-10+DDPM-6M on WRN-28-10 for $$\ell_{\infty}$$ adversary:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --configs configs/configs_cifar10.yml --exp-name EXPERIMENT_NAME --trainer madry --val-method adv --arch wrn_28_10 --activation ACTIVATION --syn-data-list diffusion_cifar10
```
For ResNet-18 use --arch resnet18.
To evaluate using AutoAttack:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py --configs configs/configs_cifar10.yml --exp-name EXPERIMENT_NAME --val-method auto --arch wrn_28_10 --activation ACTIVATION --ckpt PATH_TO_CHECKPOINT_FILE
```

Options for ACTIVATION: relu, prelu, pelu, elu, pprelu (corresponds to PReLU+ in paper), pblu (corresponds to ReBLU in paper), softplus, psoftplus, silu, psilu, pssilu (with fixed parameter $$\beta$$), pssilu2 (with learnable parameter $$\beta$$).  We note that pssilu2 corresponds to the PSSiLU results in the main paper.
