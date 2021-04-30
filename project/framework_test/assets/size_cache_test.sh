#!/bin/bash

run_test_for_res() {
  convert 2_1_sq_layout.png -filter point -resize "$1" "./$test_dir/$1_2_1_sq_layout.png"
#  cp 2_1_sq_layout.png "./$test_dir/$1_2_1_sq_layout.png"
  ./sim_cuda makeinput "./$test_dir/$1_2_1_sq_layout.png" $2 $3 --interpolate-pressure 0 --constant-velocity 1 "./$test_dir/$1_2_1_sq_layout.bin"
  ./test.sh ./fluid_accurate.json "./$test_dir/$1_2_1_sq_layout.bin" "./$test_dir"
#  ./sim_cuda fixedtime ./fluid_accurate.json --backend=cpu "./$test_dir/$1_2_1_sq_layout.bin" 10 -o "./$test_dir/test_$1_2_1_cpu.bin"
}

test_dir="$1"
if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

mkdir -p "$test_dir"
rm -f "$test_dir*"
cp 2_1_sq_layout.png "$test_dir"
run_test_for_res 260x130 20 10
run_test_for_res 380x190 29.2 14.6
run_test_for_res 460x230 35.4 17.7
run_test_for_res 520x260 40 20
run_test_for_res 600x300 46.2 23.1
run_test_for_res 640x320 49.2 24.6
run_test_for_res 700x350 53.8 26.9
run_test_for_res 740x370 56.9 28.5
run_test_for_res 840x420 64.6 32.3
run_test_for_res 1180x590 90.8 45.4
run_test_for_res 1680x840 129.2 64.6
run_test_for_res 2360x1180 181.5 90.8
run_test_for_res 4096x2048 315 157.5
