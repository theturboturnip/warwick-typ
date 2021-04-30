#!/bin/bash

FLUID_JSON=$(basename "$1")
INPUT=$(basename "$2")
TEST_DIR="$3"

if [ $# -lt 3 ]; then
    echo "Not enough arguments provided"
    exit 1
fi

mkdir -p $TEST_DIR

for backend in "cuda" "cpu";
do
    for seconds in 25;
    do
        output_dat_hash=$(sed 's/\./_/g' <<< "$FLUID_JSON.$INPUT.$seconds.$backend")
        out_file="./tests/${output_dat_hash}.txt"
        
        for i in $(seq 1 5);
        do
            echo "$backend $FLUID_JSON $INPUT $seconds"
            start=`date +%s`
            ./sim_cuda fixedtime "$1" "$2" "$seconds" --backend "$backend" --max-freq 30 -o "$TEST_DIR/bin_${output_dat_hash}_${i}.bin" >> "$out_file"
            end=`date +%s`
            echo "$backend $FLUID_JSON $INPUT simming $seconds took ~ $((end - start)) seconds."
        done
    done
done
