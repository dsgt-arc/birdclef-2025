#!/bin/bash
#SBATCH --job-name=birdclef-student-teacher-prep --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=24G
#SBATCH --qos=embers --time=1:00:00 --output=logs/Report-student-teacher-prep-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL --mail-user=acmiyaguchi@gatech.edu

set -eu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

set -x
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

version=${1:-"mel2vec-v2"}
python -m birdclef.mel2vec.student_teacher generate-token-perch \
    --tokenizer-path $scratch_dir/2025v2/$version/tokenizer_pca/n_clusters=16383 \
    --output-path $scratch_dir/2025v2/$version/soundscape-token-perch
