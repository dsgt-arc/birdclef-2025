#!/bin/bash
#SBATCH --job-name=birdclef-compile --account=paceship-dsgt_clef2025 --qos=embers
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --time=01:00:00 --output=logs/Report-compile-%j.log

set -eu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

set -x
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.kaggle.compile compile-model \
    $project_dir/models/2025/v1/$1/torch-linear-v1 $1
