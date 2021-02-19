find_package(Vulkan REQUIRED)
if (Vulkan_GLSLC_EXECUTABLE)
    message("Found glslc at ${Vulkan_GLSLC_EXECUTABLE}")
    set(GLSLC "${Vulkan_GLSLC_EXECUTABLE}")
else()
    set(GLSLC "glslc")
    message(WARNING "Vulkan package did not find glslc. Selecting '${GLSLC}'")
endif()

message(INFO "BINARY DIR IS ${PROJECT_BINARY_DIR}")
set(PROJECT_SHADER_DIR "${PROJECT_BINARY_DIR}/shaders")
add_custom_command(
        OUTPUT ${PROJECT_SHADER_DIR}
        COMMAND mkdir -p ${PROJECT_SHADER_DIR}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

function(compile_glsl out_var)
    set(result)
    foreach(in_f ${ARGN})
        get_filename_component(in_f_abs ${in_f} ABSOLUTE)
        file(RELATIVE_PATH out_f ${CMAKE_CURRENT_SOURCE_DIR} ${in_f_abs})
        set(out_f "${PROJECT_SHADER_DIR}/${out_f}.spv")
        add_custom_command(OUTPUT ${out_f}
                COMMAND ${GLSLC} -o ${out_f} ${in_f_abs}
                DEPENDS ${in_f_abs} ${CMAKE_CURRENT_SOURCE_DIR}/global_structures.glsl ${CMAKE_CURRENT_SOURCE_DIR}/global_descriptor_sets.glsl
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                COMMENT "Creating SPIR-V file ${out_f} from ${in_f}"
                VERBATIM
                IMPLICIT_DEPENDS
                )
        list(APPEND result ${out_f})
    endforeach()
    set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

compile_glsl(shader_files
        quantity_scalar.vert
        quantity_scalar.frag
        compute_sim_data_image.comp
        compute_update_particles.comp
        particle.vert
        particle.frag
        )
add_library(shaders OBJECT ${PROJECT_SHADER_DIR} ${shader_files})
