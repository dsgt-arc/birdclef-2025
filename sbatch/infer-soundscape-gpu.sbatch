#!/bin/bash
#SBATCH --job-name=infer --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --gres=gpu:1 --constraint=RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH -t120 -qembers -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=acheung@gatech.edu
set -eux
hostname

nvidia-smi
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

# manually build the necessary sif files via scripts/build-apptainer.sh
apptainer exec \
    --writable-tmpfs \
    --cleanenv \
    --nv \
    --cwd ~/scratch/birdclef/models \
    ~/scratch/birdclef/app.sif \
    bash << EOF
source ~/scratch/birdclef/app/.venv/bin/activate
set -eux
python -m birdclef.infer.workflow process-audio \
    $scratch_dir/raw/birdclef-2025/train_soundscapes \
    $scratch_dir/2025/infer-soundscape \
    --assert-gpu \
    --model-name ${1:-"BirdNET"} \
    --num-workers ${NUM_WORKERS:-1} \
    $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi)
EOF
