/**
 * @file test_covariance_matrix.cpp
 * @brief Unit tests for CovarianceMatrix C++ wrapper
 */

#include <gtest/gtest.h>
#include <stonesoup/stonesoup.hpp>
#include <cmath>

using namespace stonesoup;

class CovarianceMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(CovarianceMatrixTest, CreateSquare) {
    CovarianceMatrix m(4);
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 4);
}

TEST_F(CovarianceMatrixTest, CreateRectangular) {
    CovarianceMatrix m(3, 3);
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 3);
}

TEST_F(CovarianceMatrixTest, Identity) {
    CovarianceMatrix m = CovarianceMatrix::identity(3);

    // Diagonal should be 1
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(m(i, i), 1.0);
    }

    // Off-diagonal should be 0
    EXPECT_DOUBLE_EQ(m(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(m(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(m(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(m(2, 1), 0.0);
}

TEST_F(CovarianceMatrixTest, Diagonal) {
    std::vector<double> diag = {1.0, 2.0, 3.0, 4.0};
    CovarianceMatrix m = CovarianceMatrix::diagonal(diag);

    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 4);

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(m(i, i), diag[i]);
    }

    // Off-diagonal should be 0
    EXPECT_DOUBLE_EQ(m(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m(1, 2), 0.0);
}

TEST_F(CovarianceMatrixTest, CopyConstructor) {
    CovarianceMatrix m1 = CovarianceMatrix::identity(3);
    m1(0, 1) = 0.5;  // Make it non-identity

    CovarianceMatrix m2(m1);

    EXPECT_EQ(m2.rows(), m1.rows());
    EXPECT_EQ(m2.cols(), m1.cols());
    EXPECT_DOUBLE_EQ(m2(0, 1), 0.5);

    // Modify original, copy should be independent
    m1(0, 1) = 100.0;
    EXPECT_DOUBLE_EQ(m2(0, 1), 0.5);
}

TEST_F(CovarianceMatrixTest, MoveConstructor) {
    CovarianceMatrix m1 = CovarianceMatrix::diagonal({1.0, 2.0});
    CovarianceMatrix m2(std::move(m1));

    EXPECT_EQ(m2.rows(), 2);
    EXPECT_DOUBLE_EQ(m2(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m2(1, 1), 2.0);
}

TEST_F(CovarianceMatrixTest, CopyAssignment) {
    CovarianceMatrix m1 = CovarianceMatrix::identity(2);
    CovarianceMatrix m2(3);

    m2 = m1;

    EXPECT_EQ(m2.rows(), 2);
    EXPECT_EQ(m2.cols(), 2);
}

TEST_F(CovarianceMatrixTest, MoveAssignment) {
    CovarianceMatrix m1 = CovarianceMatrix::identity(2);
    CovarianceMatrix m2(3);

    m2 = std::move(m1);

    EXPECT_EQ(m2.rows(), 2);
    EXPECT_DOUBLE_EQ(m2(0, 0), 1.0);
}

TEST_F(CovarianceMatrixTest, ElementAccess) {
    CovarianceMatrix m(3);

    // Set elements
    m(0, 0) = 1.0;
    m(0, 1) = 0.5;
    m(1, 0) = 0.5;
    m(1, 1) = 2.0;
    m(2, 2) = 3.0;

    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 0.5);
    EXPECT_DOUBLE_EQ(m(1, 0), 0.5);
    EXPECT_DOUBLE_EQ(m(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(2, 2), 3.0);
}

TEST_F(CovarianceMatrixTest, AtThrowsOutOfRange) {
    CovarianceMatrix m(2);

    EXPECT_THROW(m.at(2, 0), std::out_of_range);
    EXPECT_THROW(m.at(0, 2), std::out_of_range);
    EXPECT_THROW(m.at(2, 2), std::out_of_range);
}

TEST_F(CovarianceMatrixTest, SetIdentity) {
    CovarianceMatrix m(3);

    // Fill with non-zero values
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            m(i, j) = 5.0;
        }
    }

    // Set to identity
    m.set_identity();

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(m(i, j), expected);
        }
    }
}

TEST_F(CovarianceMatrixTest, DataPointer) {
    CovarianceMatrix m = CovarianceMatrix::identity(2);

    double* data = m.data();
    EXPECT_NE(data, nullptr);

    // Row-major storage: [m(0,0), m(0,1), m(1,0), m(1,1)]
    EXPECT_DOUBLE_EQ(data[0], 1.0);  // m(0,0)
    EXPECT_DOUBLE_EQ(data[1], 0.0);  // m(0,1)
    EXPECT_DOUBLE_EQ(data[2], 0.0);  // m(1,0)
    EXPECT_DOUBLE_EQ(data[3], 1.0);  // m(1,1)
}
