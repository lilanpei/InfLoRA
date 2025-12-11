#!/bin/bash
#SBATCH --job-name=inflora_imgr50
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_imagenetr50_%j.out
#SBATCH --error=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_imagenetr50_%j.err

# InfLoRA on ImageNet-R (50 tasks, 4 classes per task)

cd /leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA

mkdir -p data
if [ ! -e data/imagenet-r ]; then
    ln -s ~/datasets/imagenet-r data/imagenet-r
fi

~/venv_inc_lora/bin/python main.py --device 0 --config configs/mimg50_inflora_seed42.json
