#!/bin/bash
#SBATCH --job-name=infer --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=24 --mem-per-cpu=8G
#SBATCH -t240 -qinferno -oReport-%j.out
set -eu
source ~/scratch/birdclef/.venv/bin/activate

set -x
# in case we want to SSH into the instance
hostname
# directory to share models
cd ~/scratch/birdclef/models

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.infer.workflow process-audio \
    $project_dir/raw/birdclef-2025/train_soundscapes \
    $project_dir/2025/infer-soundscape-cpu \
    --model-name ${1:-"BirdNET"} \
    --num-workers ${NUM_WORKERS:-24} \
    $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi)
