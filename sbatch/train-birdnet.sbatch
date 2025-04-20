#!/bin/bash
#SBATCH --job-name=train-birdnet --account=paceship-dsgt_clef2025
##SBATCH --nodes=1 --gres=gpu:V100:1 --cpus-per-task=12 --mem-per-gpu=96G
#SBATCH --nodes=1 --gres=gpu:1 --constraint=RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH --time=2:00:00 --qos=embers
#SBATCH --output=Report-%j.log --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
set -ue
source ~/scratch/birdclef/.venv/bin/activate
echo "Running on $(hostname)"
srun hostname

set -x

nvidia-smi
# check that python can use cuda
python -c "import torch; print(torch.cuda.is_available())"
python -c "import lightning as pl; pl.Trainer(accelerator='cuda')"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# start monitoring NVIDIA GPU metrics
NVIDIA_LOG_FILE=${SLURM_SUBMIT_DIR}/Report-${SLURM_JOB_ID}-nvidia-logs.ndjson
nvidia-logs monitor $NVIDIA_LOG_FILE --interval 15 &
nvidia_logs_pid=$!

# get current timestamp in YYYYMMDD-HHMMSS format
timestamp=$(date +'%Y%m%d-%H%M%S')

audio_path=${1:-$HOME/shared/birdclef/raw/birdclef-2025/train_audio}
metadata_path=${2:-$HOME/shared/birdclef/raw/birdclef-2025/train.csv}
output_path=${3:-$HOME/scratch/birdclef/models/birdnet/$timestamp}
mkdir -p $output_path

# python -m birdclef.model.birdnet.train \
birdclef model birdnet train \
    $audio_path \
    $metadata_path \
    $output_path \
    --num-workers 6 \
    --batch-size 100 \
    --max-epochs 2

# parse NVIDIA GPU metrics
kill $nvidia_logs_pid
nvidia-logs parse $NVIDIA_LOG_FILE
