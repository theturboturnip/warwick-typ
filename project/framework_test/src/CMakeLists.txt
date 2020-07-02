add_library(generic_sources INTERFACE)
target_sources(generic_sources INTERFACE
        cmd_line/CommandLineConverters.h
        cmd_line/CommandLineConverters.cpp
        cmd_line/CommandLineParser.cpp
        cmd_line/CommandLineParser.h
        cmd_line/subapps/ISubApp.h
        cmd_line/subapps/MakeInputSubApp.cpp
        cmd_line/subapps/MakeInputSubApp.h
        cmd_line/subapps/FixedTimeSimSubApp.cpp
        cmd_line/subapps/FixedTimeSimSubApp.h
        simulation/backends/cpu_simple/CpuOptimizedSimBackend.cpp
        simulation/backends/cpu_simple/CpuOptimizedSimBackend.h
        simulation/backends/cpu_simple/CpuSimBackendBase.cpp
        simulation/backends/cpu_simple/CpuSimBackendBase.h
        simulation/backends/cpu_simple/CpuSimpleSimBackend.cpp
        simulation/backends/cpu_simple/CpuSimpleSimBackend.h
        simulation/backends/null/NullSimulation.cpp
        simulation/backends/null/NullSimulation.h
        simulation/file_format/LegacySimDump.cpp
        simulation/file_format/LegacySimDump.h
        simulation/runners
        simulation/runners/sim_10s_runner
        simulation/runners/sim_10s_runner/ISim10sRunner.cpp
        simulation/runners/sim_10s_runner/ISim10sRunner.h
        simulation/runners/sim_10s_runner/Sim10sRunner.inl
        simulation/runners/sim_ticked_runner
        simulation/runners/sim_ticked_runner/ISimTickedRunner.cpp
        simulation/runners/sim_ticked_runner/ISimTickedRunner.h
        simulation/runners/sim_ticked_runner/SimTickedRunner.inl
        simulation/SimulationBackendEnum.h
        simulation/SimulationParameters.h
        util/fatal_error.cpp
        util/fatal_error.h
        util/HostVisibleArray.h
        util/LegacyCompat2DBackingArray.h
        validation/SimDumpDifferenceData.cpp
        validation/SimDumpDifferenceData.h
        main.cpp
        )

add_library(cuda_sources INTERFACE)
target_sources(cuda_sources INTERFACE
        simulation/backends/cuda/cuda_memory_wrappers.cu
        simulation/backends/cuda/cuda_memory_wrappers.cuh
        simulation/backends/cuda/cuda_red_black_pair.h
        simulation/backends/cuda/cuda_sim.h
        simulation/backends/cuda/cuda_vulkan_memory_wrappers.h
        simulation/backends/cuda/interface.h
        )
