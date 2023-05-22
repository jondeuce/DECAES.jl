#!/bin/bash

revisions="v0.5.0,master"
script="bench_epg.jl"
output_dir="results"
mkdir -p $output_dir

benchpkg DECAES \
    --rev=$revisions \
    --script=$script \
    --exeflags="--threads=auto -O3" \
    --output-dir=$output_dir

benchpkgplot DECAES \
    --rev=$revisions \
    --format=pdf \
    --npart=1000000 \
    --output-dir=$output_dir \
    --input-dir=$output_dir

benchpkgtable DECAES \
    --rev=$revisions \
    --input-dir=$output_dir > $output_dir/table_DECAES.txt
