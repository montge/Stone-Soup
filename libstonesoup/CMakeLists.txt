cmake_minimum_required(VERSION 3.16)

project(stonesoup
    VERSION 0.1.0
    DESCRIPTION "Stone Soup C Library - Target Tracking and State Estimation"
    LANGUAGES C
)

# Set C standard
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build documentation" OFF)
option(ENABLE_FUZZING "Enable fuzzing targets with libFuzzer" OFF)
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

# Position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(STONESOUP_SOURCES
    src/types.c
    src/matrix.c
    src/kalman.c
    src/particle.c
    src/coordinates.c
    src/frame_policies.c
)

# Header files
set(STONESOUP_HEADERS
    include/stonesoup/stonesoup.h
    include/stonesoup/types.h
    include/stonesoup/matrix.h
    include/stonesoup/kalman.h
    include/stonesoup/particle.h
    include/stonesoup/coordinates.h
    include/stonesoup/precision.h
    include/stonesoup/frame_policies.h
    include/stonesoup/version.h
)

# Shared library
add_library(stonesoup_shared SHARED ${STONESOUP_SOURCES})
set_target_properties(stonesoup_shared PROPERTIES
    OUTPUT_NAME stonesoup
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${STONESOUP_HEADERS}"
)
target_compile_definitions(stonesoup_shared PRIVATE STONESOUP_BUILD_SHARED)
target_include_directories(stonesoup_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Static library
add_library(stonesoup_static STATIC ${STONESOUP_SOURCES})
set_target_properties(stonesoup_static PROPERTIES
    OUTPUT_NAME stonesoup_static
    PUBLIC_HEADER "${STONESOUP_HEADERS}"
)
target_include_directories(stonesoup_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link math library if needed
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(stonesoup_shared PUBLIC ${MATH_LIBRARY})
    target_link_libraries(stonesoup_static PUBLIC ${MATH_LIBRARY})
endif()

# Compiler warnings
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(stonesoup_shared PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(stonesoup_static PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(stonesoup_shared PRIVATE /W4)
    target_compile_options(stonesoup_static PRIVATE /W4)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Fuzzing with libFuzzer
if(ENABLE_FUZZING)
    add_subdirectory(fuzz)
endif()

# CUDA GPU Acceleration
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")

        # Find CUDA toolkit
        find_package(CUDAToolkit REQUIRED)

        # CUDA source files
        set(STONESOUP_CUDA_SOURCES
            src/cuda/cuda_matrix.cu
            src/cuda/cuda_kalman.cu
            src/cuda/cuda_utils.cu
        )

        # CUDA library
        add_library(stonesoup_cuda SHARED ${STONESOUP_CUDA_SOURCES})
        set_target_properties(stonesoup_cuda PROPERTIES
            OUTPUT_NAME stonesoup_cuda
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
        )
        target_link_libraries(stonesoup_cuda
            PUBLIC stonesoup_shared
            PRIVATE CUDA::cudart CUDA::cublas
        )
        target_include_directories(stonesoup_cuda PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        )

        # Add CUDA define
        target_compile_definitions(stonesoup_shared PUBLIC STONESOUP_HAS_CUDA)
        target_compile_definitions(stonesoup_static PUBLIC STONESOUP_HAS_CUDA)
        target_compile_definitions(stonesoup_cuda PRIVATE STONESOUP_BUILD_CUDA)

        # Install CUDA library
        install(TARGETS stonesoup_cuda
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    else()
        message(WARNING "CUDA compiler not found, CUDA support disabled")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# Static Analysis with cppcheck (MISRA C:2012 compliance)
option(ENABLE_CPPCHECK "Enable cppcheck static analysis" OFF)
if(ENABLE_CPPCHECK)
    find_program(CPPCHECK_EXECUTABLE cppcheck)
    if(CPPCHECK_EXECUTABLE)
        message(STATUS "Found cppcheck: ${CPPCHECK_EXECUTABLE}")

        # Create custom target for MISRA analysis
        add_custom_target(cppcheck
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --std=c17
                --suppress=missingIncludeSystem
                --suppress=unusedFunction
                --error-exitcode=1
                --inline-suppr
                -I ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/src
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running cppcheck static analysis"
        )

        # MISRA C:2012 specific target (requires cppcheck premium or addon)
        add_custom_target(misra
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --std=c17
                --addon=misra
                --suppress=missingIncludeSystem
                --error-exitcode=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/src
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running MISRA C:2012 compliance check"
        )
    else()
        message(WARNING "cppcheck not found, static analysis targets disabled")
    endif()
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS stonesoup_shared stonesoup_static
    EXPORT stonesoup-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stonesoup
)

# Export targets
install(EXPORT stonesoup-targets
    FILE stonesoup-targets.cmake
    NAMESPACE stonesoup::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stonesoup
)

# Config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/stonesoup-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/stonesoup-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stonesoup
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/stonesoup-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/stonesoup-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/stonesoup-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stonesoup
)
