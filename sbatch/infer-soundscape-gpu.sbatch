#!/bin/bash
#SBATCH --job-name=infer --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --gres=gpu:1 --constraint=RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH -t240 -qinferno -oReport-%j.out
set -eux
hostname

# manually build the necessary sif files via scripts/build-apptainer.sh
apptainer exec \
    --writable-tmpfs \
    --cleanenv \
    --nv \
    --cwd ~/scratch/birdclef/models \
    ~/scratch/birdclef/app.sif \
    bash << 'EOF'
source ~/scratch/birdclef/app/.venv/bin/activate
set -eux
nvidia-smi

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

python -m birdclef.infer.workflow process-audio \
    $project_dir/raw/birdclef-2025/train_soundscapes \
    $project_dir/2025/infer-soundscape \
    --assert-gpu \
    --model-name ${1:-"BirdNET"} \
    --num-workers ${NUM_WORKERS:-1} \
    $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi)
EOF
