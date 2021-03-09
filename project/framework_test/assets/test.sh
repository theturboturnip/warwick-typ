#!/bin/bash

FLUID_JSON=$(basename "$1")
INPUT=$(basename "$2")

mkdir -p ./tests/

for backend in "cuda" "cpu_old_fast";
do
    for seconds in 10 25 50;
    do
        output_dat_hash=$(sed 's/\./_/g' <<< "$FLUID_JSON.$INPUT.$seconds.$backend")
        out_file="./tests/${output_dat_hash}.txt"
        
        for i in $(seq 1 5);
        do
            start=`date +%s`
            ./sim_cuda fixedtime "$FLUID_JSON" "$INPUT" "$seconds" --backend "$backend" -o "./tests/bin_${output_dat_hash}_${i}.bin" >> "$out_file"
            end=`date +%s`
            echo "$BACKEND simming $seconds took ~ $((end - start)) seconds."
        done
    done
done
