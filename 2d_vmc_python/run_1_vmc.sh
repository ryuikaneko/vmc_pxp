#!/bin/bash

PROG=code_vmc_2d.py

source ${HOME}/miniconda3/etc/profile.d/conda.sh ; conda activate
conda activate ml

export OMP_NUM_THREADS=1

for L in \
4
do

python ${PROG} \
  --Lx ${L} \
  --Ly ${L} \
  --jx 1.0 \
  --jn 0.0 \
  --jp 0.0 \
  --z 0.65 \
  --iterations 300 \
  --output-root ./dat \
  --verbose

done

conda deactivate
