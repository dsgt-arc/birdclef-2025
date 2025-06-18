#!/bin/bash
#SBATCH --job-name=ranasierraecnn-train        # Job name
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

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)
dataset_name=train_audio-infer-soundscape
model_name=${1:-"RanaSierraeCNN"}
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
