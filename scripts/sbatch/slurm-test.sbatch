#!/bin/bash
#SBATCH --job-name=pytest --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t20 -qinferno -oReport-%j.out
set -ue
source $SLURM_SUBMIT_DIR/scripts/activate

set +x
cd $TMPDIR
cp -r $SLURM_SUBMIT_DIR/tests .
export PYSPARK_DRIVER_MEMORY=1g
pytest -vvv tests/
