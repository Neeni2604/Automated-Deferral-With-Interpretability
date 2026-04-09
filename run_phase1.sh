#!/bin/bash
#SBATCH --job-name=sae-phase1
#SBATCH --output=slurm-phase1-%j.out
#SBATCH --error=slurm-phase1-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=granite-gpu-guest
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest
 
eval "$(conda shell.bash hook)"
conda activate deferral
cd $SLURM_SUBMIT_DIR
python phase1_main.py

