#!/bin/bash
#SBATCH --job-name=inflora_imgr10
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_imagenetr10_%j.out
#SBATCH --error=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_imagenetr10_%j.err

# InfLoRA on ImageNet-R (10 tasks, 20 classes per task)
# Expected: ~76.5% final average accuracy

cd /leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA

# Create data directory and symlink to existing ImageNet-R
mkdir -p data
if [ ! -e data/imagenet-r ]; then
    ln -s ~/datasets/imagenet-r data/imagenet-r
fi

# Run InfLoRA on ImageNet-R (10 tasks)
# Run with single seed (42) for faster comparison
~/venv_inc_lora/bin/python main.py --device 0 --config configs/mimg10_inflora_seed42.json

# To run with all 5 seeds (original paper setting):
# python main.py --device 0 --config configs/mimg10_inflora.json
