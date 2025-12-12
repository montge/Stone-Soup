# Test configuration for Stone Soup C library

# Find or build a simple test framework
# For now, we'll use a simple main-based approach

# Test executable
add_executable(test_types test_types.c)

# Link against the shared library
target_link_libraries(test_types PRIVATE stonesoup_shared)

# Add test
add_test(NAME test_types COMMAND test_types)

# Set test properties
set_tests_properties(test_types PROPERTIES
    TIMEOUT 30
    LABELS "unit"
)

# Add more tests as they are created
# add_executable(test_kalman test_kalman.c)
# target_link_libraries(test_kalman PRIVATE stonesoup_shared)
# add_test(NAME test_kalman COMMAND test_kalman)

# add_executable(test_particle test_particle.c)
# target_link_libraries(test_particle PRIVATE stonesoup_shared)
# add_test(NAME test_particle COMMAND test_particle)

# add_executable(test_coordinates test_coordinates.c)
# target_link_libraries(test_coordinates PRIVATE stonesoup_shared)
# add_test(NAME test_coordinates COMMAND test_coordinates)
