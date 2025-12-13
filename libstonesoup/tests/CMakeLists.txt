# Test configuration for Stone Soup C library

# Find or build a simple test framework
# For now, we'll use a simple main-based approach

# Test executable for types
add_executable(test_types test_types.c)
target_link_libraries(test_types PRIVATE stonesoup_shared)
add_test(NAME test_types COMMAND test_types)
set_tests_properties(test_types PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

# Test executable for matrix operations
add_executable(test_matrix test_matrix.c)
target_link_libraries(test_matrix PRIVATE stonesoup_shared)
add_test(NAME test_matrix COMMAND test_matrix)
set_tests_properties(test_matrix PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

# Test executable for Kalman filter
add_executable(test_kalman test_kalman.c)
target_link_libraries(test_kalman PRIVATE stonesoup_shared)
add_test(NAME test_kalman COMMAND test_kalman)
set_tests_properties(test_kalman PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

# Test executable for particle filter
add_executable(test_particle test_particle.c)
target_link_libraries(test_particle PRIVATE stonesoup_shared)
add_test(NAME test_particle COMMAND test_particle)
set_tests_properties(test_particle PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

# Test executable for coordinate transformations and precision
add_executable(test_coordinates test_coordinates.c)
target_link_libraries(test_coordinates PRIVATE stonesoup_shared)
add_test(NAME test_coordinates COMMAND test_coordinates)
set_tests_properties(test_coordinates PROPERTIES
    TIMEOUT 30
    LABELS "unit;precision"
)

# Test executable for frame policies
add_executable(test_frame_policies test_frame_policies.c)
target_link_libraries(test_frame_policies PRIVATE stonesoup_shared)
add_test(NAME test_frame_policies COMMAND test_frame_policies)
set_tests_properties(test_frame_policies PROPERTIES
    TIMEOUT 30
    LABELS "unit;precision"
)

# Add precision-specific test configurations
# Test with single precision
add_test(NAME test_coordinates_single_precision
    COMMAND ${CMAKE_COMMAND}
        -DSTONESOUP_PRECISION_SINGLE=ON
        -DCMAKE_BUILD_TYPE=Debug
        -B ${CMAKE_BINARY_DIR}/test_single
        -S ${CMAKE_SOURCE_DIR}
)

# Test with extended precision
add_test(NAME test_coordinates_extended_precision
    COMMAND ${CMAKE_COMMAND}
        -DSTONESOUP_PRECISION_EXTENDED=ON
        -DCMAKE_BUILD_TYPE=Debug
        -B ${CMAKE_BINARY_DIR}/test_extended
        -S ${CMAKE_SOURCE_DIR}
)
