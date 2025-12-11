#!/bin/bash
#SBATCH --job-name=inflora_multi_cif_imgr_cub
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_multi_cif_imgr_cub_%j.out
#SBATCH --error=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_multi_cif_imgr_cub_%j.err

# InfLoRA multi-dataset: CIFAR-100, ImageNet-R, CUB-200 (10,10,10 tasks)

cd /leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA

# Ensure local data directories and symlinks exist
mkdir -p data

# CIFAR-100: expect torchvision-style data under data/
# (should already be available from previous InfLoRA runs)

# ImageNet-R symlink
if [ ! -e data/imagenet-r ]; then
    ln -s ~/datasets/imagenet-r data/imagenet-r
fi

# CUB-200-2011 symlink (expects pre-split train/test under this root)
if [ ! -e data/cub ]; then
    ln -s ~/datasets/CUB_200_2011 data/cub
fi

# Launch InfLoRA multi-dataset run
~/venv_inc_lora/bin/python main.py --device 0 --config configs/multi_cifar_imgr_cub_10_10_10_inflora_seed42_ep50.json
