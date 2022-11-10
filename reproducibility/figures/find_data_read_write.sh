#!/usr/bin/env bash

# rg ".*\.read\((.*)\)" reproducibility/figures/fig2/ -I -r '$1' | uniq > archive/fig2_h5ad_file_read.txt
# rg ".*\.write\((.*)\)" reproducibility/figures/fig2/ -I -r '$1' | uniq > archive/fig2_h5ad_file_write.txt
# diff archive/finddata/fig2_h5ad_file_read.txt archive/finddata/fig2_h5ad_file_write.txt
# mamba install -y rg # install https://github.com/BurntSushi/ripgrep

set -eoux pipefail

OUTPATH=archive/finddata
mkdir -p $OUTPATH

gen_read () {
   READ_PATH="$1"
   READ_OUTFILE="$2"
   rg -I ".*\.read\((.*)\)" "$READ_PATH" -r '$1' | uniq | sort > "$OUTPATH/$READ_OUTFILE"
}

gen_write () {
   READ_PATH="$1"
   WRITE_OUTFILE="$2"
   rg -I ".*\.write\((.*)\)" "$READ_PATH" -r '$1' | uniq | sort > "$OUTPATH/$WRITE_OUTFILE"
}

gen_folder () {
    OUTPATH="$1"
    SUBFOLDER="$2"
    FIG_READ_PATH="reproducibility/figures/$SUBFOLDER/"
    FIG_READ_FILE="$SUBFOLDER"_h5ad_file_read.txt
    FIG_WRITE_FILE="$SUBFOLDER"_h5ad_file_write.txt
    gen_read "$FIG_READ_PATH" "$FIG_READ_FILE"
    gen_write "$FIG_READ_PATH" "$FIG_WRITE_FILE"
    diff -c "$OUTPATH"/"$FIG_WRITE_FILE" "$OUTPATH"/"$FIG_READ_FILE" > "$OUTPATH"/"$SUBFOLDER"_diff.txt
}

gen_folder "$OUTPATH" "fig2"
gen_folder "$OUTPATH" "fig3"
gen_folder "$OUTPATH" "suppfig1"
gen_folder "$OUTPATH" "suppfig2"
gen_folder "$OUTPATH" "suppfig3"
gen_folder "$OUTPATH" "suppfig4"

