#!/bin/bash
#SBATCH --job-name=birdclef-mel2vec --account=paceship-dsgt_clef2025 --qos=inferno
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem-per-cpu=6G
#SBATCH --time=04:00:00 --output=logs/Report-mel2vec-%j.log

set -xeu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.mel2vec.workflow \
    $scratch_dir/2025/mfcc-soundscape/data \
    $scratch_dir/2025/mel2vec \
    --gensim-workers 24 \
    --luigi-workers 1
