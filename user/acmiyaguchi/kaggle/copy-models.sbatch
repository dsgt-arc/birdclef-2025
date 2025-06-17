#!/bin/bash
#SBATCH --job-name=gcloud-models --account=paceship-dsgt_clef2025 --qos=inferno
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --time=00:10:00 --output=logs/Report-gcloud-%j.log

gcloud storage rsync -r ~/shared/birdclef/models/2025/ \
    gs://dsgt-arc-birdclef-2025/models/
