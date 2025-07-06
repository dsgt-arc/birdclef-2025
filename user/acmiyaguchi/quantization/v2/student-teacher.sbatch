#!/bin/bash
#SBATCH --job-name=birdclef-train --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --gres=gpu:1 -C RTX6000 --cpus-per-task=6 --mem-per-gpu=64G
#SBATCH --qos=embers --time=2:00:00
#SBATCH --output=logs/Report-student-teacher-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL --mail-user=acmiyaguchi@gatech.edu

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/birdclef/.venv/bin/activate

set -xe
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

version=${1:-"mel2vec-v2"}
python -m birdclef.mel2vec.student_teacher train \
    "student-teacher" \
    --root $scratch_dir/2025v2/$version \
    --data-path $scratch_dir/2025v2/$version/soundscape-token-perch
