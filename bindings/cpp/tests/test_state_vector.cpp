/**
 * @file test_state_vector.cpp
 * @brief Unit tests for StateVector C++ wrapper
 */

#include <gtest/gtest.h>
#include <stonesoup/stonesoup.hpp>
#include "test_common.hpp"

using namespace stonesoup;
using namespace stonesoup::testing;

class StateVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize library if needed
    }
};

TEST_F(StateVectorTest, CreateWithSize) {
    StateVector sv(4);
    EXPECT_EQ(sv.size(), 4);
    // Default values should be zero
    for (std::size_t i = 0; i < sv.size(); ++i) {
        EXPECT_DOUBLE_EQ(sv[i], 0.0);
    }
}

TEST_F(StateVectorTest, CreateFromInitializerList) {
    StateVector sv{1.0, 2.0, 3.0, 4.0};
    EXPECT_EQ(sv.size(), 4);
    EXPECT_DOUBLE_EQ(sv[0], 1.0);
    EXPECT_DOUBLE_EQ(sv[1], 2.0);
    EXPECT_DOUBLE_EQ(sv[2], 3.0);
    EXPECT_DOUBLE_EQ(sv[3], 4.0);
}

TEST_F(StateVectorTest, CreateFromVector) {
    std::vector<double> data = {0.5, 1.5, 2.5};
    StateVector sv(data);
    EXPECT_EQ(sv.size(), 3);
    EXPECT_DOUBLE_EQ(sv[0], 0.5);
    EXPECT_DOUBLE_EQ(sv[1], 1.5);
    EXPECT_DOUBLE_EQ(sv[2], 2.5);
}

TEST_F(StateVectorTest, CopyConstructor) {
    StateVector sv1{1.0, 2.0, 3.0};
    StateVector sv2(sv1);

    // Use helper function from test_common.hpp
    EXPECT_TRUE(state_vectors_equal(sv1, sv2));

    // Modify original, copy should be independent
    sv1[0] = 100.0;
    EXPECT_FALSE(state_vectors_equal(sv1, sv2));
    EXPECT_DOUBLE_EQ(sv2[0], 1.0);
}

TEST_F(StateVectorTest, MoveConstructor) {
    StateVector sv1{1.0, 2.0};
    StateVector sv2(std::move(sv1));

    EXPECT_EQ(sv2.size(), 2);
    EXPECT_DOUBLE_EQ(sv2[0], 1.0);
    EXPECT_DOUBLE_EQ(sv2[1], 2.0);
}

TEST_F(StateVectorTest, CopyAssignment) {
    StateVector sv1{1.0, 2.0};
    StateVector sv2(3);

    sv2 = sv1;

    EXPECT_EQ(sv2.size(), 2);
    EXPECT_DOUBLE_EQ(sv2[0], 1.0);
}

TEST_F(StateVectorTest, MoveAssignment) {
    StateVector sv1{1.0, 2.0};
    StateVector sv2(3);

    sv2 = std::move(sv1);

    EXPECT_EQ(sv2.size(), 2);
    EXPECT_DOUBLE_EQ(sv2[0], 1.0);
}

TEST_F(StateVectorTest, ElementAccess) {
    StateVector sv{1.0, 2.0, 3.0};

    // Modify via operator[]
    sv[1] = 5.0;
    EXPECT_DOUBLE_EQ(sv[1], 5.0);

    // Access via at() with bounds checking
    EXPECT_DOUBLE_EQ(sv.at(0), 1.0);
    EXPECT_DOUBLE_EQ(sv.at(1), 5.0);
    EXPECT_DOUBLE_EQ(sv.at(2), 3.0);
}

TEST_F(StateVectorTest, AtThrowsOutOfRange) {
    StateVector sv{1.0, 2.0};
    EXPECT_THROW(sv.at(2), std::out_of_range);
    EXPECT_THROW(sv.at(100), std::out_of_range);
}

TEST_F(StateVectorTest, Fill) {
    StateVector sv(5);
    sv.fill(3.14);

    for (std::size_t i = 0; i < sv.size(); ++i) {
        EXPECT_DOUBLE_EQ(sv[i], 3.14);
    }
}

TEST_F(StateVectorTest, DataPointer) {
    StateVector sv{1.0, 2.0, 3.0};

    double* data = sv.data();
    EXPECT_NE(data, nullptr);
    EXPECT_DOUBLE_EQ(data[0], 1.0);

    // Modify via data pointer
    data[0] = 10.0;
    EXPECT_DOUBLE_EQ(sv[0], 10.0);
}

TEST_F(StateVectorTest, Iterators) {
    StateVector sv{1.0, 2.0, 3.0, 4.0};

    // Range-based for loop
    double sum = 0.0;
    for (double val : sv) {
        sum += val;
    }
    EXPECT_DOUBLE_EQ(sum, 10.0);

    // Standard algorithms
    std::fill(sv.begin(), sv.end(), 1.0);
    for (std::size_t i = 0; i < sv.size(); ++i) {
        EXPECT_DOUBLE_EQ(sv[i], 1.0);
    }
}

TEST_F(StateVectorTest, ConstIterators) {
    const StateVector sv{1.0, 2.0, 3.0};

    double sum = 0.0;
    for (auto it = sv.cbegin(); it != sv.cend(); ++it) {
        sum += *it;
    }
    EXPECT_DOUBLE_EQ(sum, 6.0);
}
