//
// Created by samuel on 28/06/2020.
//

#include "CpuSimBackendBase.h"
#include <algorithm>

LegacySimDump CpuSimBackendBase::dumpStateAsLegacy() {
    auto dump = LegacySimDump();
    dump.params = params;
    dump.u = u.getBacking();
    dump.v = v.getBacking();
    dump.p = p.getBacking();
    dump.flag = flag.getBacking();

    return dump;
}

CpuSimBackendBase::CpuSimBackendBase(const LegacySimDump& dump, float baseTimestep) :
    params(dump.params),
    imax(dump.params.imax),
    jmax(dump.params.jmax),
    xlength(dump.params.xlength),
    ylength(dump.params.ylength),
    delx(xlength/imax),
    dely(ylength/jmax),
    ibound(0), // Number of boundary squares? TODO This isn't right
    ui(1.0), // Initial X Velocity
    vi(0.0), // Initial Y Velocity
    Re(150.0), // Reynolds Number
    tau(0.5), // Safety factor for timestep control
    itermax(100), // Maximum number of iterations in SOR
    eps(0.001), // Stopping error threshold for SOR
    omega(1.7), // Relaxation parameter for SOR
    gamma(0.9), // Upwind differencing factor in PDE discretisation
    baseTimestep(baseTimestep),
    u(dump.u, imax+2, jmax+2),
    v(dump.v, imax+2, jmax+2),
    f(imax+2, jmax+2, 0.0f),
    g(imax+2, jmax+2, 0.0f),
    p(dump.p, imax+2, jmax+2),
    rhs(imax+2, jmax+2, 0.0f),
    flag(dump.flag, imax+2, jmax+2)
{}

uint32_t CpuSimBackendBase::getRequiredTimestepSubdivision(float umax, float vmax) const {
    const float delt_u = delx/umax;
    const float delt_v = dely/vmax;
    const float delt_Re = 1.0/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

    const float min_delt = tau * std::min({delt_u, delt_v, delt_Re});
    DASSERT_M(min_delt < 1.0e-10, "Minimum timestep is too small");
    uint32_t subdivision = 1;
    float timestep = baseTimestep;
    while (timestep > min_delt) {
        timestep /= 2;
        subdivision *= 2;
    }

    //fprintf(stderr, "Subdiv: %02d min_delt: %5g actual timestep: %5g\r", subdivision, min_delt, timestep);

    return subdivision;

    // del_t satisfying CFL conditions
    /*if (tau >= 1.0e-10) { // else no time stepsize control TODO: why?


        if (deltu<deltv) {
            del_t_temp = min(deltu, deltRe);
        } else {
            del_t_temp = min(deltv, deltRe);
        }
        del_t = tau * (del_t_temp); // multiply by safety factor
    }*/
}
