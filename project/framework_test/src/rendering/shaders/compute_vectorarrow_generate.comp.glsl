//#version 450
//#extension GL_ARB_separate_shader_objects : enable
//
//#include "global_descriptor_sets.glsl"
//#include "global_structures.glsl"
//
//SPEC_CONST_MAX_VECTORARROW_COUNT()
//
//PUSH_CONSTANTS(VectorArrowGenerateParams)
//
//layout(set = 0, binding = 0, std430) writeonly buffer VectorArrowData {
//    VectorArrow vectorArrowDatas[maxVectorArrowCount];
//};
//layout(set = 1, binding = 0, std430) buffer VectorArrowIndices {
//    uint vectorArrowIndexList_length;
//    uint vectorArrowIndexList[maxVectorArrowCount];
//};
//layout(set = 2, binding = 0, std430) uniform sampler2D read;