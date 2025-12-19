cmake_minimum_required(VERSION 3.16)
project(stonesoup_cpp VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build C++ unit tests" ON)
option(ENABLE_CPPCHECK "Enable cppcheck static analysis" OFF)
option(ENABLE_MISRA_CXX "Enable MISRA C++ checking" OFF)

# Find libstonesoup
find_library(STONESOUP_LIBRARY
    NAMES stonesoup
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../libstonesoup/build
        ${STONESOUP_LIB_PATH}
)

find_path(STONESOUP_INCLUDE_DIR
    NAMES stonesoup/stonesoup.h
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../libstonesoup/include
        ${STONESOUP_INCLUDE_PATH}
)

if(NOT STONESOUP_LIBRARY OR NOT STONESOUP_INCLUDE_DIR)
    message(WARNING "libstonesoup not found - some features may not build")
endif()

# Header-only library interface
add_library(stonesoup_cpp INTERFACE)
target_include_directories(stonesoup_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(STONESOUP_INCLUDE_DIR)
    target_include_directories(stonesoup_cpp INTERFACE ${STONESOUP_INCLUDE_DIR})
endif()

if(STONESOUP_LIBRARY)
    target_link_libraries(stonesoup_cpp INTERFACE ${STONESOUP_LIBRARY})
endif()

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(stonesoup_cpp INTERFACE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion
    )
elseif(MSVC)
    target_compile_options(stonesoup_cpp INTERFACE /W4)
endif()

# Static analysis with cppcheck
if(ENABLE_CPPCHECK)
    find_program(CPPCHECK_EXECUTABLE cppcheck)
    if(CPPCHECK_EXECUTABLE)
        add_custom_target(cppcheck_cpp
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --std=c++17
                --suppress=missingIncludeSystem
                --error-exitcode=1
                -I ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include
            COMMENT "Running cppcheck on C++ bindings"
        )
    endif()
endif()

# MISRA C++ checking
if(ENABLE_MISRA_CXX)
    find_program(CPPCHECK_EXECUTABLE cppcheck)
    if(CPPCHECK_EXECUTABLE)
        add_custom_target(misra_cpp
            COMMAND ${CPPCHECK_EXECUTABLE}
                --enable=all
                --std=c++17
                --addon=misra
                --suppress=missingIncludeSystem
                -I ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include
            COMMENT "Running MISRA C++ checks"
        )
    endif()
endif()

# Unit tests
if(BUILD_TESTS)
    enable_testing()

    # Try to find GoogleTest
    find_package(GTest QUIET)

    if(GTest_FOUND AND STONESOUP_LIBRARY)
        add_executable(test_cpp_bindings
            tests/test_state_vector.cpp
            tests/test_covariance_matrix.cpp
            tests/test_gaussian_state.cpp
        )
        target_link_libraries(test_cpp_bindings
            PRIVATE
            stonesoup_cpp
            GTest::gtest
            GTest::gtest_main
        )
        add_test(NAME cpp_bindings_tests COMMAND test_cpp_bindings)
    else()
        message(STATUS "GoogleTest or libstonesoup not found - skipping C++ tests")
    endif()
endif()

# Install headers
install(DIRECTORY include/stonesoup
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Export targets
install(TARGETS stonesoup_cpp
    EXPORT stonesoup_cpp-targets
)

install(EXPORT stonesoup_cpp-targets
    FILE stonesoup_cppTargets.cmake
    NAMESPACE stonesoup::
    DESTINATION lib/cmake/stonesoup_cpp
)
