# birdclef-2025

- https://www.imageclef.org/BirdCLEF2025

## quickstart

We recommend using `uv` for all packaging related commands.

```bash
pip install uv
uv --help
```

Install the pre-commit hooks for formatting code:

```bash
uv tool install pre-commit
pre-commit install
```

## activating the python environment

### PACE

If you are running on PACE, then run the following command to activate the environment:

```bash
source scripts/activate
```

This will activate the virtual environment and setup packages and cache directories.
This script is also used within sbatch jobs.

### localhost

Follow typical python packaging conventions.
Create a virtual environment and install it in editable mode.

```bash
# create a virtualenvironment
uv venv

# activate it
source .venv/bin/activate

# install the package
uv pip install -e ".[dev]"
```

## validating install

Make sure the package works as expected:

```bash
birdclef --help
```

Run the tests:

```bash
# if you are on PACE
./scripts/slurm-test

# if you are on localhost
pytest -vv tests
```
