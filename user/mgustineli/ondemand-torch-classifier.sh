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
classifier_name=torch-linear-v1

python -m birdclef.torch.workflow \
    $scratch_dir/data/2025/$dataset_name/$model_name/parts/embed/ \
    $scratch_dir/models/2025/$dataset_name/$model_name/$classifier_name \
    $model_name \
