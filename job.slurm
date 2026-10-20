#!/bin/bash
#SBATCH --job-name=hujan-mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=hasil_%j.out

module load python   # sesuaikan dengan cluster
python main.py
