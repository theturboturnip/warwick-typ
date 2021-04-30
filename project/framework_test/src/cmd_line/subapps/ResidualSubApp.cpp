//
// Created by samuel on 18/04/2021.
//

#include <simulation/file_format/SimSnapshot.h>
#include <simulation/file_format/FluidParams.h>
#include "ResidualSubApp.h"

#include "simulation/backends/original/simulation.h"
#include "simulation/backends/original/constants.h"
#include "memory/FrameAllocator.h"

ResidualSubApp::ResidualSubApp() {}

void ResidualSubApp::run() {
    auto fluid_props = FluidParams::from_file(fluidPropertiesFile);
    auto input = SimSnapshot::from_file(inputPath);

    auto imax = input.simSize.internal_pixel_size.x;
    auto jmax = input.simSize.internal_pixel_size.y;
    auto delx = input.simSize.physical_size.x;
    auto dely = input.simSize.physical_size.y;

    FrameAllocator<MType::Cpu> allocator{};
    auto u = allocator.allocate2D<float>(input.simSize.padded_pixel_size);
    u.memcpy_in(input.velocity_x);
    auto v = allocator.allocate2D<float>(input.simSize.padded_pixel_size);
    v.memcpy_in(input.velocity_y);
    auto p = allocator.allocate2D<float>(input.simSize.padded_pixel_size);
    p.memcpy_in(input.pressure);
    auto rhs = allocator.allocate2D<float>(input.simSize.padded_pixel_size);
    rhs.zero_out();
    auto flag = allocator.allocate2D<char>(input.simSize.padded_pixel_size);
    flag.memcpy_in(input.to_legacy().flag);
    auto fluidmask_vector = input.get_fluidmask();
    auto fluidmask_2d = allocator.allocate2D<unsigned int>(input.simSize.padded_pixel_size);
    fluidmask_2d.memcpy_in(fluidmask_vector);

    float delta_t = -1;
    OriginalOptimized::setTimestepInterval(&delta_t,
           imax, jmax,
           delx, dely,
           u.as_cpu(), v.as_cpu(),
           fluid_props.Re,
           fluid_props.timestep_safety
    );

    // Compute RHS assuming the tentative velocity == actual velocity, del_t = max possible
    OriginalOptimized::computeRhs(
        u.as_cpu(), v.as_cpu(), rhs.as_cpu(), flag.as_cpu(),
        imax, jmax, delta_t, delx, dely
    );

    size_t i, j;
    float** u_arr = u.as_cpu();
    float** v_arr = v.as_cpu();
    float** p_arr = p.as_cpu();
    char** flag_arr = flag.as_cpu();
    float** rhs_arr = rhs.as_cpu();
    unsigned int** fluidmask = fluidmask_2d.as_cpu();

    // Compute residual from pressure values, rhs

    // Find number of fluid squares
    int ifull = 0;//std::count(fluidmask_vector.begin(), fluidmask_vector.end(), 0xFFFFFFFF);

    // Compute L2-norm of P
    const double rdx2 = 1.0/(delx*delx);
    const double rdy2 = 1.0/(dely*dely);
    double p0 = 0.0;
    {
        // Calculate sum of squares
        for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                if (flag_arr[i][j] & C_F) {
                    double p_val = p_arr[i][j];
                    p0 += p_val * p_val;
                    ifull += 1;
                }
            }
        }

        p0 = sqrt(p0 / ifull);
        if (p0 < 0.0001) { p0 = 1.0; }
    }

    double res_stack = 0;
    for (i = 1; i <= imax; i++) {
        for (j = 1; j <= jmax; j++) {
            //if ((i+j)%2 != 0) continue;
            if (flag_arr[i][j] & C_F) {
                // only fluid cells
                float add = (fluid_E_mask(p_arr[i + 1][j] - p_arr[i][j]) -
                             fluid_W_mask(p_arr[i][j] - p_arr[i - 1][j])) *
                            rdx2 +
                            (fluid_N_mask(p_arr[i][j + 1] - p_arr[i][j]) -
                             fluid_S_mask(p_arr[i][j] - p_arr[i][j - 1])) *
                            rdy2 -
                        rhs_arr[i][j];
                res_stack += add * add;
            }
        }
    }
    res_stack = sqrt((res_stack) / ifull) / p0;

    float min_pressure = std::numeric_limits<float>::max();
    float max_pressure = std::numeric_limits<float>::lowest();
    float min_v = min_pressure;
    float max_v = max_pressure;
    float min_u = min_pressure;
    float max_u = max_pressure;
    for (i = 1; i <= imax; i++) {
        for (j = 1; j <= jmax; j++) {
            //if ((i+j)%2 != 0) continue;
            if (flag_arr[i][j] & C_F) {
                // only fluid cells
                min_u = std::min(min_u, u_arr[i][j]);
                max_u = std::max(max_u, u_arr[i][j]);

                min_v = std::min(min_v, v_arr[i][j]);
                max_v = std::max(max_v, v_arr[i][j]);

                min_pressure = std::min(min_pressure, p_arr[i][j]);
                max_pressure = std::max(max_pressure, p_arr[i][j]);
            }
        }
    }

    fprintf(stdout, "%9.7g\n", res_stack);
    fprintf(stdout, "%9.7g\n", min_u);
    fprintf(stdout, "%9.7g\n", max_u);
    fprintf(stdout, "%9.7g\n", min_v);
    fprintf(stdout, "%9.7g\n", max_v);
    fprintf(stdout, "%9.7g\n", min_pressure);
    fprintf(stdout, "%9.7g\n", max_pressure);
    fprintf(stderr, "delta_t: %f\nres_stack: %f\nifull: %d\np0: %f\npressure: %g - %g\nu: %g - %g\nv: %g - %g",
            delta_t, res_stack, ifull, p0, min_pressure, max_pressure,
            min_u, max_u, min_v, max_v);
}

void ResidualSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) {
    subcommand->add_option("fluid_properties", fluidPropertiesFile, "Fluid properties file")
            ->check(CLI::ExistingFile)
            ->required(true);
    subcommand->add_option("input_file", inputPath, "Input simulation state")
            ->check(CLI::ExistingFile)
            ->required();
}
