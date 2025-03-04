#!/bin/bash
#SBATCH --job-name=soundscape --account=paceship-dsgt_clef2025
##SBATCH -N4 -n4 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH -N32 -n32 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH -t120 -qembers -oReport-%j.out
set -ue

source ~/scratch/birdclef/.venv/bin/activate
echo "Running on $(hostname)"
srun hostname

set +x
pickle_path=~/scratch/birdclef/luigi/${SLURM_JOB_ID}/state.pickle
mkdir -p $(dirname $pickle_path)
luigid --state-path $pickle_path &
pid=$!
# wait for luigid to start by curling the scheduler API
while ! curl -s localhost:8082 > /dev/null; do sleep 1; done

audio_path=$1
intermediate_path=$2
output_path=$3

srun bash -c "
    source ~/scratch/birdclef/.venv/bin/activate && \
    set -x && \
    birdclef etl embed birdnet soundscapes \
        --total-batches ${TOTAL_BATCHES:-200} \
        $(if [ -n "${LIMIT:-}" ]; then echo "--limit $LIMIT"; fi) \
        --scheduler-host $(hostname) \
        $audio_path \
        $intermediate_path
"

kill $pid

set -x
export PYSPARK_DRIVER_MEMORY=6g
export SPARK_LOCAL_DIR=$TMPDIR/spark-tmp
birdclef etl parquet repartition \
    --num-partitions ${NUM_PARTITIONS:-16} \
    $intermediate_path'/*/*.parquet' \
    $output_path

python3 <<EOF
from birdclef.spark import get_spark
df = get_spark().read.parquet("$output_path")
df.printSchema()
df.show(n=1, vertical=True, truncate=100)
print("count", df.count())
EOF
