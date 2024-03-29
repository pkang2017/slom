# Spiking GLOM
"Spiking GLOM: Bio-inspired Architecture for Next-generation Object Recognition", Peng Kang, Srutarshi Banerjee, Henry Chopp, Aggelos Katsaggelos, Oliver Cossairt. In review.

## Setup

From the main directory run:

``pipenv install``

``pip install spikingjelly``

to install the required dependencies.

## Training on Spiking GLOM / Potential-assisted Spiking GLOM 

To run contrastive pre-training on CIFAR-10, execute:

```bash
python spiking_glom_new/main.py --flagfile spiking_glom_config_new/spiking_glom_contrast.cfg
```

After pre-training, to run supervised training on CIFAR-10, execute:

```bash
python spiking_glom_new/main.py --flagfile spiking_glom_config_new/spiking_glom_supervise.cfg
```

See ``spiking_glom_new/flags_Agglomerator_slom.py`` to check all the flag meanings.

## Training on Hybrid Spiking GLOM

To run contrastive pre-training on CIFAR-10, execute:

```bash
python hybrid_spiking_glom_new/main.py --flagfile hybrid_spiking_glom_config_new/spiking_glom_contrast.cfg
```

After pre-training, to run supervised training on CIFAR-10, execute:

```bash
python hybrid_spiking_glom_new/main.py --flagfile hybrid_spiking_glom_config_new/spiking_glom_supervise.cfg
```

## Calculate the firing rates

For Potential-assisted Spiking GLOM:

```bash
python spiking_glom_new/main_energy.py --flagfile spiking_glom_config_new/spiking_glom_energy.cfg
```

For Hybrid Spiking GLOM:

```bash
python hybrid_spiking_glom_new/main.py --flagfile hybrid_spiking_glom_config_new/spiking_glom_energy.cfg
```

## Freeze to plot

```bash
python spiking_glom_new/main.py --flagfile spiking_glom_config_new/spiking_glom_plot.cfg
```

## Pre-trained models

We provide pre-trained models [Potential-assisted Spiking GLOM and Hybrid Spiking GLOM](https://drive.google.com/drive/folders/1jiVrP2k5qW7FZhe2LGsRlJmDKwtz2GeA?usp=sharing) for ``calculate the firing rates`` and specific [5-level Potential-assisted Spiking GLOM](https://drive.google.com/drive/folders/1d5-kvyTHoWjsHxm9eXtPrrfYx8wNMqs7?usp=sharing) for ``freeze to plot``.

## Credits

- Agglomerator by [Garau et al.](https://github.com/mmlab-cv/Agglomerator)
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly) 

