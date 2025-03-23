#!/bin/bash
#SBATCH --job-name=eda --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=24 --mem-per-cpu=2G
#SBATCH -t120 -qinferno -oReport-%j.out
set -ue
export NO_REINSTALL=1
source ~/scratch/birdclef/.venv/bin/activate
birdclef eda duration
