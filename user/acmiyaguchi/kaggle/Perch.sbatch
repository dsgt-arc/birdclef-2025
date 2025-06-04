#!/bin/bash
#SBATCH --job-name=birdclef-mel2vec --account=paceship-dsgt_clef2025 --qos=inferno
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem-per-cpu=6G
#SBATCH --time=04:00:00 --output=logs/Report-Perch-%j.log

set -eu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

set -x
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.kaggle.main \
    $scratch_dir/test_soundscape \
    $scratch_dir/tmp/output \
    $project_dir/models/2025/train_audio-infer-soundscape/Perch/torch-linear-v1 \
    Perch
