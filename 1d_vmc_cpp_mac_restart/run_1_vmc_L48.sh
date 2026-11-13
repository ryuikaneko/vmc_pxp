#!/bin/bash

PROG=code_vmc_1d
L=48

/opt/local/bin/mpiexec -n 8 ${PROG} \
  --L ${L} \
  --jx 1.0 \
  --jn 0.0 \
  --jp 0.0 \
  --z 0.65 \
  --iterations 300 \
  --average-iterations 100 \
  --opt-samples 5000 \
  --final-samples 200000 \
  --output-root dat \
  --output-prefix dat01 \
  --initial-noise 1.0e-6 \
  --verbose

/opt/local/bin/mpiexec -n 8 ${PROG} \
  --L ${L} \
  --jx 1.0 \
  --jn 0.0 \
  --jp 0.0 \
  --z 0.65 \
  --iterations 300 \
  --average-iterations 100 \
  --opt-samples 5000 \
  --final-samples 200000 \
  --output-root dat \
  --output-prefix dat02 \
  --initial-noise 1.0e-6 \
  --restart dat/L${L}_jx1_jn0_jp0/dat01_checkpoint.csv \
  --verbose

/opt/local/bin/mpiexec -n 8 ${PROG} \
  --L ${L} \
  --jx 1.0 \
  --jn 0.0 \
  --jp 0.0 \
  --z 0.65 \
  --iterations 300 \
  --average-iterations 100 \
  --opt-samples 5000 \
  --final-samples 200000 \
  --output-root dat \
  --output-prefix dat03 \
  --initial-noise 1.0e-6 \
  --restart dat/L${L}_jx1_jn0_jp0/dat02_checkpoint.csv \
  --verbose
