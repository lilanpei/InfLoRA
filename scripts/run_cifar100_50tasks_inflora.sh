#!/bin/bash
#SBATCH --job-name=inflora_cifar100_t50
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_cifar100_t50_%j.out
#SBATCH --error=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_cifar100_t50_%j.err

# InfLoRA on CIFAR-100 (50 tasks, 2 classes per task)

cd /leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA

mkdir -p data
# Ensure CIFAR-100 data is available via symlink
if [ ! -e data/cifar-100-python ]; then
    ln -s ~/datasets/cifar100/cifar-100-python data/cifar-100-python
fi

~/venv_inc_lora/bin/python main.py --device 0 --config configs/cifar100_50tasks_inflora_seed42.json
