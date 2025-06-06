#!/bin/bash
#SBATCH --job-name=birdclef-perch --account=paceship-dsgt_clef2025 --qos=inferno
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --time=04:00:00 --output=logs/Report-perch-%j.log

set -eu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

set -x
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.kaggle.compile compile-perch \
    $project_dir/models/2025/v1/Perch/torch-linear-v1
