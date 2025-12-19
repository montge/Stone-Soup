/**
 * @file test_common.hpp
 * @brief Common test utilities and typed test fixtures for Stone Soup C++ bindings
 *
 * This header provides reusable test macros and fixtures to reduce code duplication
 * across test files for StateVector, CovarianceMatrix, and GaussianState.
 */

#ifndef STONESOUP_TEST_COMMON_HPP
#define STONESOUP_TEST_COMMON_HPP

#include <gtest/gtest.h>
#include <stonesoup/stonesoup.hpp>
#include <type_traits>

namespace stonesoup {
namespace testing {

/**
 * @brief Test fixture template for types with copy/move semantics
 *
 * Provides common test cases for copy constructors, move constructors,
 * copy assignment, and move assignment operators.
 */
template <typename T>
class CopyMoveTestFixture : public ::testing::Test {
protected:
    // Subclasses should implement these to create test instances
    virtual T create_instance() = 0;
    virtual T create_different_instance() = 0;
    virtual bool instances_equal(const T& a, const T& b) = 0;
};

/**
 * @brief Macro to define copy constructor test
 *
 * Tests that:
 * 1. Copy has the same values as original
 * 2. Modifications to original don't affect copy
 */
#define STONESOUP_TEST_COPY_CONSTRUCTOR(TestFixture, TypeName) \
    TEST_F(TestFixture, CopyConstructor) { \
        TypeName original = this->create_instance(); \
        TypeName copy(original); \
        EXPECT_TRUE(this->instances_equal(original, copy)); \
    }

/**
 * @brief Macro to define move constructor test
 *
 * Tests that moved-to object has expected values
 */
#define STONESOUP_TEST_MOVE_CONSTRUCTOR(TestFixture, TypeName) \
    TEST_F(TestFixture, MoveConstructor) { \
        TypeName original = this->create_instance(); \
        TypeName reference = this->create_instance(); \
        TypeName moved(std::move(original)); \
        EXPECT_TRUE(this->instances_equal(reference, moved)); \
    }

/**
 * @brief Macro to define copy assignment test
 *
 * Tests that copy assignment produces equal objects
 */
#define STONESOUP_TEST_COPY_ASSIGNMENT(TestFixture, TypeName) \
    TEST_F(TestFixture, CopyAssignment) { \
        TypeName original = this->create_instance(); \
        TypeName other = this->create_different_instance(); \
        other = original; \
        EXPECT_TRUE(this->instances_equal(original, other)); \
    }

/**
 * @brief Macro to define move assignment test
 *
 * Tests that move assignment transfers ownership correctly
 */
#define STONESOUP_TEST_MOVE_ASSIGNMENT(TestFixture, TypeName) \
    TEST_F(TestFixture, MoveAssignment) { \
        TypeName original = this->create_instance(); \
        TypeName reference = this->create_instance(); \
        TypeName other = this->create_different_instance(); \
        other = std::move(original); \
        EXPECT_TRUE(this->instances_equal(reference, other)); \
    }

/**
 * @brief Macro to define all copy/move semantics tests
 *
 * Convenience macro that defines all four copy/move tests
 */
#define STONESOUP_TEST_COPY_MOVE_SEMANTICS(TestFixture, TypeName) \
    STONESOUP_TEST_COPY_CONSTRUCTOR(TestFixture, TypeName) \
    STONESOUP_TEST_MOVE_CONSTRUCTOR(TestFixture, TypeName) \
    STONESOUP_TEST_COPY_ASSIGNMENT(TestFixture, TypeName) \
    STONESOUP_TEST_MOVE_ASSIGNMENT(TestFixture, TypeName)

/**
 * @brief Helper to compare double values with tolerance
 */
inline bool doubles_equal(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) < tolerance;
}

/**
 * @brief Helper to compare StateVector instances
 */
inline bool state_vectors_equal(const StateVector& a, const StateVector& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!doubles_equal(a[i], b[i])) return false;
    }
    return true;
}

/**
 * @brief Helper to compare CovarianceMatrix instances
 */
inline bool covariance_matrices_equal(const CovarianceMatrix& a, const CovarianceMatrix& b) {
    if (a.dim() != b.dim()) return false;
    for (std::size_t i = 0; i < a.dim(); ++i) {
        for (std::size_t j = 0; j < a.dim(); ++j) {
            if (!doubles_equal(a(i, j), b(i, j))) return false;
        }
    }
    return true;
}

/**
 * @brief Helper to compare GaussianState instances
 */
inline bool gaussian_states_equal(const GaussianState& a, const GaussianState& b) {
    if (a.dim() != b.dim()) return false;
    for (std::size_t i = 0; i < a.dim(); ++i) {
        if (!doubles_equal(a.state(i), b.state(i))) return false;
    }
    for (std::size_t i = 0; i < a.dim(); ++i) {
        for (std::size_t j = 0; j < a.dim(); ++j) {
            if (!doubles_equal(a.covar(i, j), b.covar(i, j))) return false;
        }
    }
    return doubles_equal(a.timestamp(), b.timestamp());
}

} // namespace testing
} // namespace stonesoup

#endif // STONESOUP_TEST_COMMON_HPP
