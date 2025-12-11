#!/bin/bash
#SBATCH -J idomainnet_inflora_nowncm
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH -o /leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_idomainnet_inflora_nowncm_%j.out
#SBATCH -e /leonardo/home/userexternal/lli00001/dc_inc/baselines/InfLoRA/logs/slurm_idomainnet_inflora_nowncm_%j.err

cd ~/dc_inc/baselines/InfLoRA

~/venv_inc_lora/bin/python main.py   --device 0   --config configs/idomainnet_inflora_seed42_nowncm.json
