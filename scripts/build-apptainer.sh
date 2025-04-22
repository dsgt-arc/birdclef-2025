#!/bin/bash
export APPTAINER_CACHEDIR=$(realpath ~/scratch/apptainer/cache)
mkdir -p ~/scratch/apptainer/cache
apptainer build \
    ~/scratch/birdclef/app.sif \
    ~/clef/birdclef-2025/app.def
