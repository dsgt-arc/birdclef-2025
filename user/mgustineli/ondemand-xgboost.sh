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
dataset_name=train_audio
model_name=${1:-"Perch"}
# model names:
# - BirdNET
# - YAMNet
# - Perch
# - HawkEars
# - BirdSetConvNeXT
# - BirdSetEfficientNetB1
# - RanaSierraeCNN
search_method=${2:-"random"}  # grid, random, or bayesian
pickle_name=xgboost-model.pkl

python -m birdclef.classifier.xgboost main \
    $scratch_dir/data/2025/subset-${dataset_name}-infer-soundscape-cpu \
    $scratch_dir/models/2025/subset-${dataset_name}-infer-soundscape-cpu/$pickle_name \
    --search-method $search_method \
