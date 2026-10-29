#!/bin/bash
#SBATCH --job-name=h_sg_gpu
#SBATCH --partition=hpc-high
#SBATCH --account=cw-sup
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --gres=gpu:h200:1
#SBATCH --output=/mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/logs/h_sg_gpu-%A_%a.log
python3 /mnt/home/kdidi/tmol-paper-benchmark/runs/20261029-figure1-fixed50-min/src/dispatch_sg_hivrt.py gpu results/sg
