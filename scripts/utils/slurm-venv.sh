#!/usr/bin/env bash
# usage: source scripts/slurm-venv.sh [venv_root]
SCRIPT_PARENT_ROOT=$(
    dirname ${BASH_SOURCE[0]} \
    | dirname $(cat -) \
    | realpath $(cat -)
)

# choose the module depending if this being run on slurm or not
MODULE_PATH=$(dirname $SCRIPT_PARENT_ROOT)
# MODULE_PATH=${SLURM_SUBMIT_DIR:-$MODULE_PATH}
# optional argument to specify the venv root
VENV_PARENT_ROOT=${1:-~/scratch/birdclef}
VENV_PARENT_ROOT=$(realpath $VENV_PARENT_ROOT)

# use an updated version of python and set the include path for wheels
module load python/3.10
export CPATH=$PYTHON_ROOT/include/python3.10:$CPATH

# create a virtual environment with uv
python -m ensurepip
python -m pip install --upgrade pip uv
mkdir -p $VENV_PARENT_ROOT
pushd $VENV_PARENT_ROOT

# check if exists
if [[ -d .venv ]]; then
    echo "Virtual environment already exists. Skipping creation."
    source .venv/bin/activate
else
    uv venv
    source .venv/bin/activate
fi

# check for NO_REINSTALL flag
if [[ -z ${NO_REINSTALL:-} ]]; then
    uv pip install -e $MODULE_PATH
fi
popd
