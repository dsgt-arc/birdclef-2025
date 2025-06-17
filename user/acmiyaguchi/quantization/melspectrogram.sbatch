#!/bin/bash
#SBATCH --job-name=birdclef-melspectrogram --account=paceship-dsgt_clef2025 --qos=inferno
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem-per-cpu=6G
#SBATCH --time=02:00:00  --output=logs/Report-melspectrogram-%A_%a.out

set -xeu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.mfcc.workflow \
    $project_dir/raw/birdclef-2025/train_soundscapes \
    $scratch_dir/2025/mfcc-soundscape \
    --num-workers ${NUM_WORKERS:-24} \
    $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi)
