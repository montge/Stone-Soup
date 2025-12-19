/**
 * @file test_gaussian_state.cpp
 * @brief Unit tests for GaussianState C++ wrapper
 */

#include <gtest/gtest.h>
#include <stonesoup/stonesoup.hpp>
#include "test_common.hpp"

using namespace stonesoup;
using namespace stonesoup::testing;

class GaussianStateTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(GaussianStateTest, CreateWithDimension) {
    GaussianState state(4);
    EXPECT_EQ(state.dim(), 4);
}

TEST_F(GaussianStateTest, CreateFromStateAndCovariance) {
    StateVector sv{1.0, 2.0, 3.0};
    CovarianceMatrix cov = CovarianceMatrix::identity(3);

    GaussianState state(sv, cov);

    EXPECT_EQ(state.dim(), 3);
    EXPECT_DOUBLE_EQ(state.state(0), 1.0);
    EXPECT_DOUBLE_EQ(state.state(1), 2.0);
    EXPECT_DOUBLE_EQ(state.state(2), 3.0);
}

TEST_F(GaussianStateTest, DimensionMismatchThrows) {
    StateVector sv{1.0, 2.0};
    CovarianceMatrix cov(3);  // 3x3 matrix, but state is 2D

    EXPECT_THROW(GaussianState(sv, cov), StoneSoupException);
}

TEST_F(GaussianStateTest, CopyConstructor) {
    GaussianState state1(2);
    state1.state(0) = 1.0;
    state1.state(1) = 2.0;
    state1.set_covariance_identity();

    GaussianState state2(state1);

    // Use helper function from test_common.hpp
    EXPECT_TRUE(gaussian_states_equal(state1, state2));

    // Modify original, copy should be independent
    state1.state(0) = 100.0;
    EXPECT_FALSE(gaussian_states_equal(state1, state2));
    EXPECT_DOUBLE_EQ(state2.state(0), 1.0);
}

TEST_F(GaussianStateTest, MoveConstructor) {
    GaussianState state1(2);
    state1.state(0) = 1.0;
    state1.state(1) = 2.0;

    GaussianState state2(std::move(state1));

    EXPECT_EQ(state2.dim(), 2);
    EXPECT_DOUBLE_EQ(state2.state(0), 1.0);
    EXPECT_DOUBLE_EQ(state2.state(1), 2.0);
}

TEST_F(GaussianStateTest, CopyAssignment) {
    GaussianState state1(2);
    state1.state(0) = 1.0;
    state1.state(1) = 2.0;
    state1.set_covariance_identity();

    GaussianState state2(3);
    state2 = state1;

    // Use helper function from test_common.hpp
    EXPECT_TRUE(gaussian_states_equal(state1, state2));
}

TEST_F(GaussianStateTest, MoveAssignment) {
    GaussianState state1(2);
    state1.state(0) = 1.0;

    GaussianState state2(3);
    state2 = std::move(state1);

    EXPECT_EQ(state2.dim(), 2);
    EXPECT_DOUBLE_EQ(state2.state(0), 1.0);
}

TEST_F(GaussianStateTest, StateAccess) {
    GaussianState state(4);

    // Set state values
    state.state(0) = 0.0;
    state.state(1) = 1.0;
    state.state(2) = 2.0;
    state.state(3) = 3.0;

    EXPECT_DOUBLE_EQ(state.state(0), 0.0);
    EXPECT_DOUBLE_EQ(state.state(1), 1.0);
    EXPECT_DOUBLE_EQ(state.state(2), 2.0);
    EXPECT_DOUBLE_EQ(state.state(3), 3.0);
}

TEST_F(GaussianStateTest, CovarianceAccess) {
    GaussianState state(2);
    state.set_covariance_identity();

    // Check identity
    EXPECT_DOUBLE_EQ(state.covar(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(state.covar(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(state.covar(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(state.covar(1, 1), 1.0);

    // Modify covariance
    state.covar(0, 1) = 0.5;
    state.covar(1, 0) = 0.5;

    EXPECT_DOUBLE_EQ(state.covar(0, 1), 0.5);
    EXPECT_DOUBLE_EQ(state.covar(1, 0), 0.5);
}

TEST_F(GaussianStateTest, Timestamp) {
    GaussianState state(2);

    // Default timestamp should be 0
    EXPECT_DOUBLE_EQ(state.timestamp(), 0.0);

    // Set timestamp
    state.set_timestamp(123.456);
    EXPECT_DOUBLE_EQ(state.timestamp(), 123.456);
}

TEST_F(GaussianStateTest, SetCovarianceIdentity) {
    GaussianState state(3);

    // Set some non-identity values
    state.covar(0, 0) = 5.0;
    state.covar(0, 1) = 2.0;
    state.covar(1, 1) = 3.0;

    // Set to identity
    state.set_covariance_identity();

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(state.covar(i, j), expected);
        }
    }
}

TEST_F(GaussianStateTest, TypicalUsage) {
    // Create a 4D state: [x, vx, y, vy]
    GaussianState state(4);

    // Initialize state vector
    state.state(0) = 0.0;   // x position
    state.state(1) = 10.0;  // x velocity
    state.state(2) = 100.0; // y position
    state.state(3) = -5.0;  // y velocity

    // Set initial covariance to identity
    state.set_covariance_identity();

    // Scale position uncertainty
    state.covar(0, 0) = 100.0;  // x position uncertainty
    state.covar(2, 2) = 100.0;  // y position uncertainty

    // Set timestamp
    state.set_timestamp(0.0);

    // Verify
    EXPECT_EQ(state.dim(), 4);
    EXPECT_DOUBLE_EQ(state.state(0), 0.0);
    EXPECT_DOUBLE_EQ(state.state(1), 10.0);
    EXPECT_DOUBLE_EQ(state.covar(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(state.covar(1, 1), 1.0);
}
