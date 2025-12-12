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

# TODO: Add coordinate system tests when coordinates module is ready
# add_executable(test_coordinates test_coordinates.c)
# target_link_libraries(test_coordinates PRIVATE stonesoup_shared)
# add_test(NAME test_coordinates COMMAND test_coordinates)
