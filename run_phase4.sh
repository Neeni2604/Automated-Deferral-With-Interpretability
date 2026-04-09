#!/bin/bash
#SBATCH --job-name=sae-phase4
#SBATCH --output=slurm-phase4-%j.out
#SBATCH --error=slurm-phase4-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=granite-gpu-guest
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest
 
eval "$(conda shell.bash hook)"
conda activate deferral
cd $SLURM_SUBMIT_DIR
python clustering_main.py

