//
// Created by samuel on 28/06/2020.
//

#pragma once

#include <memory>
#include "simulation/file_format/LegacySimDump.h"
#include "util/LegacyCompat2DBackingArray.h"
#include "CpuSimBackendBase.h"

class CpuOptimizedSimBackend : public CpuSimBackendBase {
public:
    float findMaxTimestep();
    int tick(float timestep);
    LegacySimDump dumpStateAsLegacy();
    SimSnapshot get_snapshot();

    class Frame : public CpuSimBackendBase::BaseFrame {
    public:
        Frame(FrameAllocator<MType::Cpu>& alloc,
              Size<uint32_t> paddedSize);

        Size<uint32_t> redBlackSize;

        Sim2DArray<float, MType::Cpu> p_beta, p_beta_red, p_beta_black;
        Sim2DArray<float, MType::Cpu> p_red;
        Sim2DArray<float, MType::Cpu> p_black;

        Sim2DArray<float, MType::Cpu> rhs_red;
        Sim2DArray<float, MType::Cpu> rhs_black;

        Sim2DArray<int, MType::Cpu> fluidmask;
        Sim2DArray<int, MType::Cpu> surroundmask_red;
        Sim2DArray<int, MType::Cpu> surroundmask_black;
    };

    explicit CpuOptimizedSimBackend(std::vector<Frame> frames, const FluidParams& params, const SimSnapshot& s);

protected:
    void resetFrame(Frame& frame, const SimSnapshot& s);

    std::vector<Frame> frames;
    int lastWrittenFrame;
};