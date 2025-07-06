#!/bin/bash
#SBATCH --job-name=birdclef-mel2vec-v3 --account=paceship-dsgt_clef2025
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=8G
#SBATCH --qos=embers --time=08:00:00 --output=logs/Report-mel2vec-v3-%j.log

set -eu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

set -x
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

export PYSPARK_EXECUTOR_CORES=${PYSPARK_EXECUTOR_CORES:-16}
export PYSPARK_DRIVER_MEMORY=$(( $PYSPARK_EXECUTOR_CORES*8-4 ))g
python -m birdclef.mel2vec.workflow evaluate-v3 \
    $scratch_dir/2025v2/mfcc-train/data \
    $scratch_dir/2025v2/mfcc-soundscape/data \
    $scratch_dir/2025v2/mel2vec-v3 \
    --gensim-workers $PYSPARK_EXECUTOR_CORES \
    --luigi-workers 1
