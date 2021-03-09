./sim_cuda fixedtime --backend cpu fluid_100.json initial_aca.bin 10 -o temp_cpu_out.bin
./sim_cuda fixedtime --backend cuda fluid_100.json initial_aca.bin 10 -o temp_cuda_out.bin
./sim_cuda compare temp_cpu_out.bin temp_cuda_out.bin