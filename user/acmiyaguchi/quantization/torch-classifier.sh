#!/bin/bash
#SBATCH --job-name=birdclef-train        # Job name
#SBATCH --account=paceship-dsgt_clef2025        # charge account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --gres=gpu:1                            # GPU resource
#SBATCH -C RTX6000                              # GPU type
#SBATCH --cpus-per-task=6                       # Number of cores per task
#SBATCH --mem-per-gpu=64G                       # Memory per core
#SBATCH --time=12:00:00                         # Duration of the job
#SBATCH --qos=inferno                           # QOS Name
#SBATCH --output=logs/Report-train-%j.log             # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=acmiyaguchi@gatech.edu  # E-mail address for notifications

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/birdclef/.venv/bin/activate

set -xe
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)
dataset_name=train_audio-infer-soundscape
model_name=${1:-"mel2vec"}

python -m birdclef.torch.workflow \
    $scratch_dir/2025/mfcc-train/data \
    $project_dir/models/2025/v2/$model_name \
    $model_name \
    --batch-size ${BATCH_SIZE:-64} \
