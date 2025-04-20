#!/bin/bash
#SBATCH --job-name=rsync --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t20 -qinferno -oReport-%j.out
set -e

project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/birdclef
scratch_dir=$(realpath ~/scratch/birdclef)
# trim trailing slash
prefix=${1%/}

# exit if prefix doesn't exist
if [ ! -d $scratch_dir/$prefix ]; then
    echo "Directory $scratch_dir/$prefix does not exist"
    exit 1
fi

# now copy directories from project into scratch
rclone sync -v --progress --transfers 12 $scratch_dir/$prefix/ $project_dir/$prefix
# now rewrite permissions
chmod -R 777 $project_dir
