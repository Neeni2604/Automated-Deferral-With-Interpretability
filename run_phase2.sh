#!/bin/bash
#SBATCH --job-name=sae-phase2
#SBATCH --output=slurm-phase2-%j.out
#SBATCH --error=slurm-phase2-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=granite-gpu-guest
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest
 
eval "$(conda shell.bash hook)"
conda activate deferral
cd $SLURM_SUBMIT_DIR
python phase2_main.py

