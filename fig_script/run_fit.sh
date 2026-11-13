#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh ; conda activate
conda activate ml

export OMP_NUM_THREADS=1

for PROG in \
code_fit_1d.py \
code_fit_2d.py \
code_wf_1d.py
#
#code_fit_1d.py
#code_fit_2d.py
#code_wf_1d.py
do

python ${PROG}

done

conda deactivate



cp fig_tikz/tikz_lualatex.pdf fig_1.pdf
cp wavefunc_plots_1d/pairing_parameters_initial_optimized.pdf fig_2.pdf
cp wavefunc_plots_1d/wavefunction_initial_optimized_exact.pdf fig_3.pdf
cp finite_size_plots_1d/jx1_jn0_jp0/combined_observables_vs_inverse_L.pdf fig_4.pdf
cp finite_size_plots_2d/jx1_jn0_jp0_aspect1/combined_observables_vs_inverse_sqrtN.pdf fig_5.pdf
