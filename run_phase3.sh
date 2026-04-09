#!/bin/bash
#SBATCH --job-name=sae-phase3
#SBATCH --output=slurm-phase3-%j.out
#SBATCH --error=slurm-phase3-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=granite-gpu-guest
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest
 
eval "$(conda shell.bash hook)"
conda activate deferral
cd $SLURM_SUBMIT_DIR
python phase3_main.py

