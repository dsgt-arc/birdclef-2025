#!/bin/bash
#SBATCH --job-name=soundscape --account=paceship-dsgt_clef2025
#SBATCH -N10 -n4 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH -t60 -qembers -oReport-%j.out
set -ue

source ~/scratch/birdclef/.venv/bin/activate
echo "Running on $(hostname)"
srun hostname

set +x
luigid &
pid=$!
# wait for luigid to start by curling the scheduler API
while ! curl -s localhost:8082 > /dev/null; do sleep 1; done

audio_path=$1
intermediate_path=$2
output_path=$3
total_batches=${TOTAL_BATCHES:-200}
num_partitions=${NUM_PARTITIONS:-16}
limit=${LIMIT:-}
srun bash -c "
    source ~/scratch/birdclef/.venv/bin/activate && \
    set -x && \
    birdclef etl embed birdnet soundscapes \
        $audio_path \
        $intermediate_path \
        $output_path \
        --total-batches $total_batches \
        --num-partitions $num_partitions \
        --limit $limit \
        --scheduler-host $(hostname)
"

kill $pid
