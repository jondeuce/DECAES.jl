#!/usr/bin/bash

hyperfine \
    --warmup 1 \
    --min-runs 3 \
    --export-markdown $2.md \
    --export-json $2.json \
    "julia -e 'using DECAES; main()' @$1"
