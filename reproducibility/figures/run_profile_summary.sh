#!/usr/bin/env bash

set -eoux pipefail

set -a
source .env
set +a

ENV_PATH="$CONDA_ENV_PATH" # set in local .env file
OUTPUT_DIR="output"

# command to list python files in figures folder
# $ find "reproducibility/figures" -name "*.py" -type f -printf "%f\n"

profile_summary () {
   FIG_PATH="$1"
   FILE_HANDLE="$2"
   OUTPUT_PATH="../$OUTPUT_DIR"
   (
      cd "$FIG_PATH"
      "$ENV_PATH"/time "$ENV_PATH"/gprof2dot -f pstats "$OUTPUT_PATH"/"$FILE_HANDLE".stats | "$ENV_PATH"/dot -Tpdf -o "$OUTPUT_PATH"/"$FILE_HANDLE".pdf
   )
}

profile_summary_stage_1 () {

   # fig 2
   ## stage 1
   profile_summary "fig2" "fig2_pancreas_data"
   profile_summary "fig2" "fig2_pbmc_data"

   # fig 3
   ## stage 1
   profile_summary "fig3" "fig3_unipotent_mono_model1"
   profile_summary "fig3" "fig3_unipotent_neu_model1"
   profile_summary "fig3" "fig3_uni_bifurcation_model2"
   profile_summary "fig3" "fig3_allcells_model2"

}

profile_summary_stage_2 () {

   # fig 2
   ## stage 2
   profile_summary "fig2" "fig2"

   # fig 3
   ## stage 2
   profile_summary "fig3" "fig3_allcells_cospar"
   profile_summary "fig3" "fig3_allcells_cytotrace"

}

profile_summary_stage_3 () {

   profile_summary "fig3" "fig3"

}


profile_summary_supp_stage_1 () {

   # profile_summary "suppfig1" "suppfig1_models_graph" # requires pyro render_model in >=1.8.1
   profile_summary "suppfig2" "suppfig2"
   profile_summary "suppfig3" "fig2_pancreas_data_model2"
   profile_summary "suppfig3" "fig2_pbmc_data_model2"
   profile_summary "suppfig4" "suppfig4_larry_cytotrace"

}

profile_summary_supp_stage_2 () {

   profile_summary "suppfig3" "fig2_model2"

}


profile_summary_stage_1
profile_summary_stage_2
profile_summary_stage_3

profile_summary_supp_stage_1
profile_summary_supp_stage_2
