#!/bin/bash
#SBATCH --job-name=infer --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=24 --mem-per-cpu=8G
#SBATCH -t240 -qinferno -oReport-%j.out
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
model_name=${1:-"BirdNET"}
# model names:
# - BirdNET
# - YAMNet
# - Perch
# - HawkEars
# - BirdSetConvNeXT
# - BirdSetEfficientNetB1
# - RanaSierraeCNN

python -m birdclef.infer.workflow process-audio \
    $project_dir/raw/birdclef-2025/$dataset_name \
    $scratch_dir/data/2025/${dataset_name}-infer-soundscape \
    --model-name $model_name \
    --num-workers ${NUM_WORKERS:-24} \
    $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi) \
    # $(if [ "${USE_SUBSET:-false}" = "true" ]; then echo "--use-subset"; fi) \
    # $(if [ -n "${SUBSET_SIZE:-}" ]; then echo "--subset-size $SUBSET_SIZE"; fi)
