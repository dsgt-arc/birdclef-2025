#!/bin/bash
set -xe

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/birdclef/.venv/bin/activate

# directory to share models
cd ~/scratch/birdclef/models

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)
dataset_name=train_audio-infer-soundscape
model_name=${1:-"Perch"}
# model names:
# - BirdNET
# - YAMNet
# - Perch
# - HawkEars
# - BirdSetConvNeXT
# - BirdSetEfficientNetB1
# - RanaSierraeCNN

python -m birdclef.torch.workflow \
    $project_dir/data/2025/$dataset_name/$model_name/parts/embed/ \
    $project_dir/models/2025/v2/$model_name \
    $model_name \
    --batch-size ${BATCH_SIZE:-64} \
