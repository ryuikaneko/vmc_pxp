#!/bin/bash
#SBATCH -J vmc_2d
#SBATCH -p i8cpu
#SBATCH -N 2
#SBATCH -n 128
#SBATCH --ntasks-per-node=64
##SBATCH -N 1
##SBATCH -n 128
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH -o vmc_2d.%j.out
#SBATCH -e vmc_2d.%j.err

L=14
PROG=code_vmc_2d

set -euo pipefail

module purge
module load oneapi_compiler/2023.0.0
module load openmpi/4.1.5-oneapi-2023.0.0-classic

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "========== job information =========="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "Number nodes : ${SLURM_NNODES}"
echo "MPI ranks    : ${SLURM_NTASKS}"
echo "CPUs/task    : ${SLURM_CPUS_PER_TASK}"
echo "Working dir  : $(pwd)"
echo "Start time   : $(date)"
echo

echo "========== loaded modules =========="
module list 2>&1
echo

echo "========== MPI environment =========="
echo "mpicxx path:"
command -v mpicxx
echo
echo "mpicxx backend:"
mpicxx -show
echo

echo "========== executable =========="
ls -lh ${PROG}
file ${PROG}
echo

if ldd ${PROG} | grep -q "not found"; then
    echo "ERROR: missing shared libraries" >&2
    ldd ${PROG} >&2
    exit 1
fi

echo "Shared-library check: OK"
echo

echo "========== calculation =========="

srun ${PROG} \
  --Lx ${L} \
  --Ly ${L} \
  --jx 1.0 \
  --jn 0.0 \
  --jp 0.0 \
  --z 0.65 \
  --iterations 100 \
  --average-iterations 100 \
  --opt-samples 10240 \
  --final-samples 204800 \
  --output-root dat \
  --output-prefix dat25 \
  --initial-noise 1.0e-6 \
  --restart dat/Lx${L}_Ly${L}_jx1_jn0_jp0/dat24_checkpoint.csv \
  --verbose

status=$?

echo
echo "========== finished =========="
echo "End time : $(date)"
echo "Exit code: ${status}"

exit "${status}"
