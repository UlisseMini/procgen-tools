#!/bin/bash

# Exit on any error
set -e

# Signal to notebooks that they are being run in a CI environment where they should run faster
export CI="true"

# Find all Jupyter notebooks in the repository
find . -name "*.ipynb" -not -path '*/\.*/*' -print0 \
  | xargs -0 -P 4 -rI '{}' sh -c 'jupyter nbconvert --to notebook --execute --inplace "{}" 2>&1 | tee "{}-log.txt"; echo "Finished executing {}"'

echo "All notebooks executed successfully."

# TODO:
# - Make ./experiments/channel_55.ipynb faster
