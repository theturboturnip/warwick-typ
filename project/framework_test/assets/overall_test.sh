#!/bin/bash

./test.sh ./fluid_100.json ./initial_aca.bin
./test.sh ./fluid_200.json ./initial_aca.bin
./test.sh ./fluid_300.json ./initial_aca.bin
./test.sh ./fluid_accurate.json ./initial_aca.bin
./test.sh ./fluid_100.json ./initial_circles_large.bin 
./test.sh ./fluid_200.json ./initial_circles_large.bin 
./test.sh ./fluid_300.json ./initial_circles_large.bin 
./test.sh ./fluid_accurate.json ./initial_circles_large.bin
