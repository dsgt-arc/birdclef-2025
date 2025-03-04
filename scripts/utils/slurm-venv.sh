#!/usr/bin/env bash
SCRIPT_PARENT_ROOT=$(
    dirname ${BASH_SOURCE[0]} \
    | dirname $(cat -) \
    | realpath $(cat -)
)

VENV_PARENT_ROOT=${1:-~/scratch/birdclef}
VENV_PARENT_ROOT=$(realpath $VENV_PARENT_ROOT)

# use an updated version of python and set the include path for wheels
module load python/3.10
PYTHON_ROOT=$(python -c 'import sys; print(sys.base_prefix)')
export CPATH=$PYTHON_ROOT/include/python3.10:$CPATH

# create a virtual environment with uv
# check pip and uv are installed
if ! command -v pip &> /dev/null; then python -m ensurepip; fi
if ! command -v uv &> /dev/null; then python -m pip install --upgrade pip uv; fi
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
    uv pip install -e $(dirname $SCRIPT_PARENT_ROOT)
fi
popd
