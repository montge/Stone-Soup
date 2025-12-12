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

# Position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(STONESOUP_SOURCES
    src/types.c
    src/kalman.c
    src/particle.c
    src/coordinates.c
)

# Header files
set(STONESOUP_HEADERS
    include/stonesoup/stonesoup.h
    include/stonesoup/types.h
    include/stonesoup/kalman.h
    include/stonesoup/particle.h
    include/stonesoup/coordinates.h
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
