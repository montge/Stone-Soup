// Stone Soup State Vector Unit Tests

// Load Stone Soup module
exec(get_absolute_file_path("test_state_vector.tst") + "../../loader.sce", -1);

// Test: Create zero state vector
sv = StateVector(4);
assert_checkequal(size(sv), [4, 1]);
assert_checkequal(sv, [0; 0; 0; 0]);

// Test: Create filled state vector
sv = StateVector(3, 1.5);
assert_checkequal(sv, [1.5; 1.5; 1.5]);

// Test: Create from data
sv = StateVector([1; 2; 3; 4]);
assert_checkequal(sv, [1; 2; 3; 4]);

// Test: State vector norm
sv = StateVector([3; 4]);
n = sv_norm(sv);
assert_checkalmostequal(n, 5.0, 1e-10);

// Test: State vector addition
sv1 = StateVector([1; 2; 3]);
sv2 = StateVector([4; 5; 6]);
result = sv_add(sv1, sv2);
assert_checkequal(result, [5; 7; 9]);

// Test: State vector subtraction
result = sv_subtract(sv1, sv2);
assert_checkequal(result, [-3; -3; -3]);

// Test: State vector scaling
result = sv_scale(sv1, 2.0);
assert_checkequal(result, [2; 4; 6]);

disp("All state vector tests passed!");
