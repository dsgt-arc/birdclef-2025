#!/bin/bash
#SBATCH --job-name=infer --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --gres=gpu:1 --constraint=RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH -t240 -qinferno -oReport-%j.out
module load python/3.10.10 cudnn/8.5.0.96-11.7-cuda
module list

set -eu
source ~/scratch/birdclef/.venv/bin/activate
set -x
hostname
nvidia-smi
# python -c "import os; print(f'Python Exec: {os.sys.executable}'); print(f'LD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\")}'); print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\")}'); import tensorflow as tf; print(f'TF Version: {tf.__version__}'); print(tf.config.list_physical_devices('GPU'))"

cd ~/scratch/birdclef/models

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.infer.workflow process-audio \
    $project_dir/raw/birdclef-2025/train_soundscapes \
    $project_dir/2025/infer-soundscape \
    --assert-gpu \
    --model-name ${1:-"BirdNET"} \
    --num-workers ${NUM_WORKERS:-1} \
    $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi)
