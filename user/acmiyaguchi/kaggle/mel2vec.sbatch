#!/bin/bash
#SBATCH --job-name=birdclef-mel2vec --account=paceship-dsgt_clef2025 --qos=inferno
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --time=04:00:00 --output=logs/Report-mel2vec-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL --mail-user=acmiyaguchi@gatech.edu
set -eu

hostname
source ~/scratch/birdclef/.venv/bin/activate
echo $PWD

set -x
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)

# copy over some files to the main model directory
model_dir=$project_dir/models/2025/v2-anthony/mel2vec
cp $scratch_dir/2025/mel2vec-v1/word2vec/tokenizer=tokenizer/tokenizer_n_clusters=16383/vector_size=384/window=80/ns_exponent=1.5/sample=1e-05/epochs=100/word2vec.wordvectors $model_dir/
cp $scratch_dir/2025/mel2vec-v1/tokenizer/n_clusters=16383/centroids.npy $model_dir/

python -m birdclef.kaggle.main \
    $scratch_dir/test_soundscape \
    $scratch_dir/tmp/mel2vec \
    $project_dir/models/2025/v2-anthony/mel2vec \
    mel2vec
