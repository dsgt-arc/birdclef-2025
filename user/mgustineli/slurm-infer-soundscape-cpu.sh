#!/bin/bash
#SBATCH --job-name=birdsetefficientnetb1-infer         # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --gres=gpu:1                            # GPU resource
#SBATCH -C RTX6000                              # GPU type
#SBATCH --cpus-per-task=6                       # Number of cores per task
#SBATCH --mem-per-gpu=64G                       # Memory per core
#SBATCH --time=12:00:00                         # Duration of the job
#SBATCH --qos=inferno                           # QOS Name
#SBATCH --output=Report-%j.log                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=murilogustineli@gatech.edu  # E-mail address for notifications
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
model_name=${1:-"BirdSetEfficientNetB1"}
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

# CPU usage:
# SBATCH --job-name=ranasierraecnn-infer --account=paceship-dsgt_clef2025
# SBATCH -N1 -n1 --cpus-per-task=24 --mem-per-cpu=8G
# SBATCH -t240 -qinferno -oReport-%j.out
