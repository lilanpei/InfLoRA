#!/bin/bash
#SBATCH --job-name=inflora_cifar100
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_cifar100_%j.out
#SBATCH --error=/leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_cifar100_%j.err

# InfLoRA on CIFAR-100 (10 tasks, 10 classes per task)
# Expected: ~86.5% final average accuracy

cd /leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA

# Create data directory if needed
mkdir -p data

# Run InfLoRA on CIFAR-100
# Run with single seed (42) for faster comparison
~/venv_inc_lora/bin/python main.py --device 0 --config configs/cifar100_inflora_seed42.json

# To run with all 5 seeds (original paper setting):
# python main.py --device 0 --config configs/cifar100_inflora.json
