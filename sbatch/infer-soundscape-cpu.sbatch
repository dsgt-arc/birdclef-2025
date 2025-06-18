#!/bin/bash
#SBATCH --job-name=infer-BirdNET --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=16 --mem-per-cpu=8G
#SBATCH -t120 -qembers -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acheung@gatech.edu
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
    $scratch_dir/raw/birdclef-2025/train_soundscapes \
    $scratch_dir/2025/infer-soundscape \
    --model-name ${1:-"BirdNET"} \
    --num-workers ${NUM_WORKERS:-24} \
    $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi) \
    $(if [ "${USE_SUBSET:-false}" = "true" ]; then echo "--use-subset"; fi) \
    $(if [ -n "${SUBSET_SIZE:-}" ]; then echo "--subset-size $SUBSET_SIZE"; fi)
