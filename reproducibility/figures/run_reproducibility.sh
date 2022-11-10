#!/usr/bin/env bash

set -eoux pipefail

# example command for running a single
# python module
#
# $ cd fig2/model1
# $ nohup python -m cProfile \
#     -o ../../output/fig2_pancreas_data.stats \
#     fig2_pancreas_data.py > ../../output/fig2_pancreas_data.log 2>&1 &

set -a
source .env
set +a

ENV_PATH="$CONDA_ENV_PATH"
OUTPUT_DIR="output"

run_fig () {
   FIG_PATH="$1"
   FILE_HANDLE="$2"
   OUTPUT_PATH="../$OUTPUT_DIR"
   (
      cd "$FIG_PATH"
      "$ENV_PATH"/time nohup "$ENV_PATH"/python -m cProfile \
        -o "$OUTPUT_PATH"/"$FILE_HANDLE".stats \
        "$FILE_HANDLE".py > "$OUTPUT_PATH"/"$FILE_HANDLE".log 2>&1 &
   )
}


# command to list python files in figures folder
# $ find "reproducibility/figures" -name "*.py" -type f -printf "%f\n"

main_fig_stage_1 () {

   # fig 2
   ## stage 1
   run_fig "fig2/model1" "fig2_pancreas_data"
   run_fig "fig2/model1" "fig2_pbmc_data"

   # fig 3
   ## stage 1
   run_fig "fig3" "fig3_unipotent_mono_model1"
   run_fig "fig3" "fig3_unipotent_neu_model1"
   run_fig "fig3" "fig3_uni_bifurcation_model2"
   run_fig "fig3" "fig3_allcells_model2"

}

main_fig_stage_2 () {

   # fig 2
   ## stage 2
   run_fig "fig2/model1" "fig2"

   # fig 3
   ## stage 2
   run_fig "fig3" "fig3_allcells_cospar"
   run_fig "fig3" "fig3_allcells_cytotrace"

}

main_fig_stage_3 () {

   run_fig "fig3" "fig3"

}


supp_fig_stage_1 () {

   # # run_fig "suppfig1" "suppfig1_models_graph" # requires pyro render_model in >=1.8.1
   run_fig "suppfig2" "suppfig2"
   run_fig "suppfig3/model2" "fig2_pancreas_data_model2"
   run_fig "suppfig3/model2" "fig2_pbmc_data_model2"
   run_fig "suppfig4" "suppfig4_larry_cytotrace"

}

supp_fig_stage_2 () {

   run_fig "suppfig3/model2" "fig2_model2"

}


main_fig_stage_1
main_fig_stage_2
main_fig_stage_3

supp_fig_stage_1
supp_fig_stage_2
