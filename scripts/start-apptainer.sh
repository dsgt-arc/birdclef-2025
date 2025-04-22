#!/bin/bash
echo "Starting Apptainer container, run the following command to enter the container:"
echo "source ~/scratch/birdclef/app/.venv/bin/activate"
apptainer exec \
    --writable-tmpfs \
    --cleanenv \
    --nv \
    --cwd ~/scratch/birdclef/models \
    ~/scratch/birdclef/app.sif \
    bash --noprofile --norc
