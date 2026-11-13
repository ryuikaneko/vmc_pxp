#!/bin/bash

PROG=code_make_latex_table.py

source ${HOME}/miniconda3/etc/profile.d/conda.sh ; conda activate
conda activate ml

export OMP_NUM_THREADS=1

python ${PROG} ./dat --output-dir ./fit

conda deactivate
