//
// Created by samuel on 02/07/2020.
//

#include "LegacyRenderPPMSubApp.h"

#include "simulation/file_format/LegacySimDump.h"
#include "util/LegacyCompat2DBackingArray.h"

void LegacyRenderPPMSubApp::run() {
    auto dump = LegacySimDump::fromFile(inputFile);

    FILE* fout = fopen(outputFile.c_str(), "wb");
    FATAL_ERROR_IF(fout == nullptr, "Could not open '%s'\n", outputFile.c_str());

    const int imax = dump.params.imax, jmax = dump.params.jmax;
    const float delx = dump.params.xlength/imax;
    const float dely = dump.params.ylength/jmax;
    const auto u = LegacyCompat2DBackingArray<float>(dump.u, imax+2, jmax+2);
    const auto v = LegacyCompat2DBackingArray<float>(dump.v, imax+2, jmax+2);
    const auto p = LegacyCompat2DBackingArray<float>(dump.p, imax+2, jmax+2);
    const auto flag = LegacyCompat2DBackingArray<uint8_t>(dump.flag, imax+2, jmax+2);

    const float pressure_max = *(std::max_element(dump.p.begin(), dump.p.end()));

    auto zeta = LegacyCompat2DBackingArray<float>(imax+2, jmax+2, 0.0f);
    auto psi = LegacyCompat2DBackingArray<float>(imax+2, jmax+2, 0.0f);

    int i, j;

    // Computation of the vorticity zeta at the upper right corner
    // of cell (i,j) (only if the corner is surrounded by fluid cells)
    for (i=1;i<=imax-1;i++) {
        for (j=1;j<=jmax-1;j++) {
            if ( (flag[i][j] & C_F) && (flag[i+1][j] & C_F) &&
                 (flag[i][j+1] & C_F) && (flag[i+1][j+1] & C_F)) {
                zeta[i][j] = (u[i][j+1]-u[i][j])/dely
                             -(v[i+1][j]-v[i][j])/delx;
            } else {
                zeta[i][j] = 0.0;
            }
        }
    }

    // Computation of the stream function at the upper right corner
    // of cell (i,j) (only if bother lower cells are fluid cells)
    for (i=0;i<=imax;i++) {
        psi[i][0] = 0.0;
        for (j=1;j<=jmax;j++) {
            psi[i][j] = psi[i][j-1];
            if ((flag[i][j] & C_F) || (flag[i+1][j] & C_F)) {
                psi[i][j] += u[i][j]*dely;
            }
        }
    }

    fprintf(fout, "P6 %d %d 255\n", (int)imax, (int)jmax);

    for (j = 1; j < jmax+1 ; j++) {
        for (i = 1; i < imax+1 ; i++) {
            // Initialize colors to magenta so that it's clear if something has gone wrong
            int r = 255, g = 0, b = 255;
            if (!(flag[i][j] & C_F)) {
                r = 0; b = 0; g = 255;
            } else {
                /*zmax = max(zmax, zeta[i][j]);
                zmin = min(zmin, zeta[i][j]);
                pmax = max(pmax, psi[i][j]);
                pmin = min(pmin, psi[i][j]);
                umax = max(umax, u[i][j]);
                umin = min(umin, u[i][j]);
                vmax = max(vmax, v[i][j]);
                vmin = min(vmin, v[i][j]);*/
                if (outputMode == OutputMode::Zeta) {
                    float z = (i < imax && j < jmax)?zeta[i][j]:0.0;
                    r = g = b = pow(fabs(z/12.6),.4) * 255;
                } else if (outputMode == OutputMode::Psi) {
                    float rendered_psi = (i < imax && j < jmax)?psi[i][j]:0.0;
                    r = g = b = (rendered_psi+3.0)/7.5 * 255;
                } else if (outputMode == OutputMode::Pressure) {
                    //fprintf(stderr, "i:%d, j:%d, p: %f\n",i,j,p[i][j]);
                    r = g = b = ((i < imax && j < jmax)?p[i][j]:0.0) * 255 /pressure_max;
                } else {
                    FATAL_ERROR("Unsupported output mode %d", outputMode);
                }
            }
            fprintf(fout, "%c%c%c", r, g, b);
        }
    }

    fclose(fout);
}

void LegacyRenderPPMSubApp::setupArgumentsForSubcommand(CLI::App *subcommand, const CommandLineConverters &converters) {
    subcommand->add_option("input_file", inputFile, "Simulation .bin file to render")
        ->check(CLI::ExistingFile)
        ->required();

    std::map<std::string, OutputMode> outputModeMap = {
            {"psi", OutputMode::Psi},
            {"zeta", OutputMode::Zeta},
            {"pressure", OutputMode::Pressure},
    };
    subcommand->add_option("mode", outputMode, "Type of data to render")
            ->required()
            ->transform(ENUM_TRANSFORMER(outputModeMap));

    subcommand->add_option("output_file", outputFile, "Output .ppm file to render into")
        ->required();
}
