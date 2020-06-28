//
// Created by samuel on 28/06/2020.
//

#include "CpuSimBackendBase.h"

LegacySimDump CpuSimBackendBase::dumpStateAsLegacy() {
    auto dump = LegacySimDump();
    dump.params = params;
    dump.u = u.getBacking();
    dump.v = v.getBacking();
    dump.p = p.getBacking();
    dump.flag = flag.getBacking();

    return dump;
}

CpuSimBackendBase::CpuSimBackendBase(const LegacySimDump& dump) :
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
    del_t(0.003),
    u(dump.u, imax+2, jmax+2),
    v(dump.v, imax+2, jmax+2),
    f(imax+2, jmax+2, 0.0f),
    g(imax+2, jmax+2, 0.0f),
    p(dump.p, imax+2, jmax+2),
    rhs(imax+2, jmax+2, 0.0f),
    flag(dump.flag, imax+2, jmax+2)
{}