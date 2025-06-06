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
dataset_name=subset-train_audio-infer-soundscape-cpu
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
pickle_name=xgboost-model-${search_method}-v1.pkl

python -m birdclef.classifier.xgboost \
    $scratch_dir/data/2025/$dataset_name/$model_name/parts/embed/ \
    $scratch_dir/models/2025/$dataset_name/$model_name/$pickle_name \
    --search-method $search_method \
