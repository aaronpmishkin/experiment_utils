#!/bin/bash
#
#SBATCH --job-name=gpu
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=24G

ml reset
ml restore scnn

eval "$JOB_STR -I $SLURM_ARRAY_TASK_ID"
