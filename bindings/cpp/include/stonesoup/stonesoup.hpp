/**
 * @file stonesoup.hpp
 * @brief Modern C++ bindings for Stone Soup tracking library
 *
 * This header provides RAII wrappers for the Stone Soup C library types,
 * offering exception-safe, modern C++ interfaces.
 *
 * @example
 * ```cpp
 * #include <stonesoup/stonesoup.hpp>
 *
 * using namespace stonesoup;
 *
 * // Create a 4D Gaussian state
 * GaussianState state(4);
 * state.state_vector()[0] = 0.0;  // x position
 * state.state_vector()[1] = 1.0;  // x velocity
 * state.set_covariance_identity();
 *
 * // Kalman prediction
 * auto predicted = kalman::predict(state, F, Q);
 * ```
 */

#ifndef STONESOUP_HPP
#define STONESOUP_HPP

#include <memory>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <string>
#include <optional>

// Include the C library headers
extern "C" {
#include <stonesoup/stonesoup.h>
}

namespace stonesoup {

/**
 * @brief Exception class for Stone Soup errors
 */
class StoneSoupException : public std::runtime_error {
public:
    explicit StoneSoupException(stonesoup_error_t error)
        : std::runtime_error(stonesoup_error_string(error))
        , error_code_(error) {}

    explicit StoneSoupException(const std::string& message)
        : std::runtime_error(message)
        , error_code_(STONESOUP_ERROR_INVALID_ARG) {}

    [[nodiscard]] stonesoup_error_t error_code() const noexcept {
        return error_code_;
    }

private:
    stonesoup_error_t error_code_;
};

/**
 * @brief Check error code and throw if not success
 */
inline void check_error(stonesoup_error_t error) {
    if (error != STONESOUP_SUCCESS) {
        throw StoneSoupException(error);
    }
}

/**
 * @brief RAII wrapper for state vectors
 *
 * StateVector provides exception-safe memory management for state vectors,
 * with convenient accessor methods and arithmetic operations.
 */
class StateVector {
public:
    /**
     * @brief Create a zero-initialized state vector
     * @param size Number of elements
     * @throws StoneSoupException on allocation failure
     */
    explicit StateVector(std::size_t size) {
        ptr_ = stonesoup_state_vector_create(size);
        if (!ptr_) {
            throw StoneSoupException(STONESOUP_ERROR_ALLOCATION);
        }
    }

    /**
     * @brief Create from initializer list
     * @param init Initial values
     */
    StateVector(std::initializer_list<double> init)
        : StateVector(init.size()) {
        std::copy(init.begin(), init.end(), ptr_->data);
    }

    /**
     * @brief Create from vector
     * @param data Initial values
     */
    explicit StateVector(const std::vector<double>& data)
        : StateVector(data.size()) {
        std::copy(data.begin(), data.end(), ptr_->data);
    }

    /**
     * @brief Copy constructor
     */
    StateVector(const StateVector& other) {
        ptr_ = stonesoup_state_vector_copy(other.ptr_);
        if (!ptr_) {
            throw StoneSoupException(STONESOUP_ERROR_ALLOCATION);
        }
    }

    /**
     * @brief Move constructor
     */
    StateVector(StateVector&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    /**
     * @brief Destructor
     */
    ~StateVector() {
        if (ptr_) {
            stonesoup_state_vector_free(ptr_);
        }
    }

    /**
     * @brief Copy assignment
     */
    StateVector& operator=(const StateVector& other) {
        if (this != &other) {
            StateVector tmp(other);
            std::swap(ptr_, tmp.ptr_);
        }
        return *this;
    }

    /**
     * @brief Move assignment
     */
    StateVector& operator=(StateVector&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                stonesoup_state_vector_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get size of vector
     */
    [[nodiscard]] std::size_t size() const noexcept {
        return ptr_ ? ptr_->size : 0;
    }

    /**
     * @brief Element access
     */
    [[nodiscard]] double& operator[](std::size_t i) {
        return ptr_->data[i];
    }

    /**
     * @brief Element access (const)
     */
    [[nodiscard]] const double& operator[](std::size_t i) const {
        return ptr_->data[i];
    }

    /**
     * @brief Element access with bounds checking
     */
    [[nodiscard]] double& at(std::size_t i) {
        if (i >= size()) {
            throw std::out_of_range("StateVector index out of range");
        }
        return ptr_->data[i];
    }

    /**
     * @brief Element access with bounds checking (const)
     */
    [[nodiscard]] const double& at(std::size_t i) const {
        if (i >= size()) {
            throw std::out_of_range("StateVector index out of range");
        }
        return ptr_->data[i];
    }

    /**
     * @brief Get raw data pointer
     */
    [[nodiscard]] double* data() noexcept {
        return ptr_->data;
    }

    /**
     * @brief Get raw data pointer (const)
     */
    [[nodiscard]] const double* data() const noexcept {
        return ptr_->data;
    }

    /**
     * @brief Fill with value
     */
    void fill(double value) {
        check_error(stonesoup_state_vector_fill(ptr_, value));
    }

    /**
     * @brief Get underlying C pointer (for interop)
     */
    [[nodiscard]] stonesoup_state_vector_t* get() noexcept {
        return ptr_;
    }

    [[nodiscard]] const stonesoup_state_vector_t* get() const noexcept {
        return ptr_;
    }

    // Iterator support
    using iterator = double*;
    using const_iterator = const double*;

    iterator begin() noexcept { return ptr_->data; }
    iterator end() noexcept { return ptr_->data + size(); }
    [[nodiscard]] const_iterator begin() const noexcept { return ptr_->data; }
    [[nodiscard]] const_iterator end() const noexcept { return ptr_->data + size(); }
    [[nodiscard]] const_iterator cbegin() const noexcept { return ptr_->data; }
    [[nodiscard]] const_iterator cend() const noexcept { return ptr_->data + size(); }

private:
    stonesoup_state_vector_t* ptr_ = nullptr;
};

/**
 * @brief RAII wrapper for covariance matrices
 */
class CovarianceMatrix {
public:
    /**
     * @brief Create a zero-initialized covariance matrix
     * @param rows Number of rows
     * @param cols Number of columns
     */
    CovarianceMatrix(std::size_t rows, std::size_t cols) {
        ptr_ = stonesoup_covariance_matrix_create(rows, cols);
        if (!ptr_) {
            throw StoneSoupException(STONESOUP_ERROR_ALLOCATION);
        }
    }

    /**
     * @brief Create a square covariance matrix
     * @param dim Dimension (rows = cols)
     */
    explicit CovarianceMatrix(std::size_t dim)
        : CovarianceMatrix(dim, dim) {}

    /**
     * @brief Create an identity matrix
     * @param dim Dimension
     */
    static CovarianceMatrix identity(std::size_t dim) {
        CovarianceMatrix m(dim);
        m.set_identity();
        return m;
    }

    /**
     * @brief Create a diagonal matrix
     * @param diag Diagonal values
     */
    static CovarianceMatrix diagonal(const std::vector<double>& diag) {
        CovarianceMatrix m(diag.size());
        for (std::size_t i = 0; i < diag.size(); ++i) {
            m(i, i) = diag[i];
        }
        return m;
    }

    /**
     * @brief Copy constructor
     */
    CovarianceMatrix(const CovarianceMatrix& other) {
        ptr_ = stonesoup_covariance_matrix_copy(other.ptr_);
        if (!ptr_) {
            throw StoneSoupException(STONESOUP_ERROR_ALLOCATION);
        }
    }

    /**
     * @brief Move constructor
     */
    CovarianceMatrix(CovarianceMatrix&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    /**
     * @brief Destructor
     */
    ~CovarianceMatrix() {
        if (ptr_) {
            stonesoup_covariance_matrix_free(ptr_);
        }
    }

    /**
     * @brief Copy assignment
     */
    CovarianceMatrix& operator=(const CovarianceMatrix& other) {
        if (this != &other) {
            CovarianceMatrix tmp(other);
            std::swap(ptr_, tmp.ptr_);
        }
        return *this;
    }

    /**
     * @brief Move assignment
     */
    CovarianceMatrix& operator=(CovarianceMatrix&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                stonesoup_covariance_matrix_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] std::size_t rows() const noexcept { return ptr_ ? ptr_->rows : 0; }
    [[nodiscard]] std::size_t cols() const noexcept { return ptr_ ? ptr_->cols : 0; }

    /**
     * @brief Get dimension (for square matrices, returns rows)
     */
    [[nodiscard]] std::size_t dim() const noexcept { return rows(); }

    /**
     * @brief Element access (row, col)
     */
    [[nodiscard]] double& operator()(std::size_t row, std::size_t col) {
        return ptr_->data[row * ptr_->cols + col];
    }

    [[nodiscard]] const double& operator()(std::size_t row, std::size_t col) const {
        return ptr_->data[row * ptr_->cols + col];
    }

    /**
     * @brief Element access with bounds checking
     */
    [[nodiscard]] double& at(std::size_t row, std::size_t col) {
        if (row >= rows() || col >= cols()) {
            throw std::out_of_range("CovarianceMatrix index out of range");
        }
        return (*this)(row, col);
    }

    [[nodiscard]] const double& at(std::size_t row, std::size_t col) const {
        if (row >= rows() || col >= cols()) {
            throw std::out_of_range("CovarianceMatrix index out of range");
        }
        return (*this)(row, col);
    }

    /**
     * @brief Set to identity matrix
     */
    void set_identity() {
        check_error(stonesoup_covariance_matrix_eye(ptr_));
    }

    /**
     * @brief Get raw data pointer
     */
    [[nodiscard]] double* data() noexcept { return ptr_->data; }
    [[nodiscard]] const double* data() const noexcept { return ptr_->data; }

    /**
     * @brief Get underlying C pointer
     */
    [[nodiscard]] stonesoup_covariance_matrix_t* get() noexcept { return ptr_; }
    [[nodiscard]] const stonesoup_covariance_matrix_t* get() const noexcept { return ptr_; }

private:
    stonesoup_covariance_matrix_t* ptr_ = nullptr;
};

/**
 * @brief RAII wrapper for Gaussian states
 */
class GaussianState {
public:
    /**
     * @brief Create a Gaussian state
     * @param state_dim Dimension of state vector
     */
    explicit GaussianState(std::size_t state_dim) {
        ptr_ = stonesoup_gaussian_state_create(state_dim);
        if (!ptr_) {
            throw StoneSoupException(STONESOUP_ERROR_ALLOCATION);
        }
    }

    /**
     * @brief Create from state vector and covariance
     */
    GaussianState(const StateVector& state, const CovarianceMatrix& covar)
        : GaussianState(state.size()) {
        if (state.size() != covar.rows() || state.size() != covar.cols()) {
            throw StoneSoupException("Dimension mismatch between state and covariance");
        }
        std::copy(state.begin(), state.end(), ptr_->state_vector->data);
        std::copy(covar.data(), covar.data() + covar.rows() * covar.cols(),
                  ptr_->covariance->data);
    }

    /**
     * @brief Copy constructor
     */
    GaussianState(const GaussianState& other) {
        ptr_ = stonesoup_gaussian_state_copy(other.ptr_);
        if (!ptr_) {
            throw StoneSoupException(STONESOUP_ERROR_ALLOCATION);
        }
    }

    /**
     * @brief Move constructor
     */
    GaussianState(GaussianState&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    /**
     * @brief Destructor
     */
    ~GaussianState() {
        if (ptr_) {
            stonesoup_gaussian_state_free(ptr_);
        }
    }

    /**
     * @brief Copy assignment
     */
    GaussianState& operator=(const GaussianState& other) {
        if (this != &other) {
            GaussianState tmp(other);
            std::swap(ptr_, tmp.ptr_);
        }
        return *this;
    }

    /**
     * @brief Move assignment
     */
    GaussianState& operator=(GaussianState&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                stonesoup_gaussian_state_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get state dimension
     */
    [[nodiscard]] std::size_t dim() const noexcept {
        return ptr_ && ptr_->state_vector ? ptr_->state_vector->size : 0;
    }

    /**
     * @brief Access state vector element
     */
    [[nodiscard]] double& state(std::size_t i) {
        return ptr_->state_vector->data[i];
    }

    [[nodiscard]] const double& state(std::size_t i) const {
        return ptr_->state_vector->data[i];
    }

    /**
     * @brief Access covariance element
     */
    [[nodiscard]] double& covar(std::size_t row, std::size_t col) {
        return ptr_->covariance->data[row * ptr_->covariance->cols + col];
    }

    [[nodiscard]] const double& covar(std::size_t row, std::size_t col) const {
        return ptr_->covariance->data[row * ptr_->covariance->cols + col];
    }

    /**
     * @brief Get/set timestamp
     */
    [[nodiscard]] double timestamp() const noexcept { return ptr_->timestamp; }
    void set_timestamp(double t) noexcept { ptr_->timestamp = t; }

    /**
     * @brief Set covariance to identity
     */
    void set_covariance_identity() {
        check_error(stonesoup_covariance_matrix_eye(ptr_->covariance));
    }

    /**
     * @brief Get underlying C pointer
     */
    [[nodiscard]] stonesoup_gaussian_state_t* get() noexcept { return ptr_; }
    [[nodiscard]] const stonesoup_gaussian_state_t* get() const noexcept { return ptr_; }

private:
    stonesoup_gaussian_state_t* ptr_ = nullptr;
};

/**
 * @brief RAII wrapper for particle states
 */
class ParticleState {
public:
    /**
     * @brief Create a particle state
     * @param num_particles Number of particles
     * @param state_dim Dimension of state vector per particle
     */
    ParticleState(std::size_t num_particles, std::size_t state_dim) {
        ptr_ = stonesoup_particle_state_create(num_particles, state_dim);
        if (!ptr_) {
            throw StoneSoupException(STONESOUP_ERROR_ALLOCATION);
        }
    }

    /**
     * @brief Copy constructor
     */
    ParticleState(const ParticleState& other) = delete;  // Not implemented in C API

    /**
     * @brief Move constructor
     */
    ParticleState(ParticleState&& other) noexcept
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    /**
     * @brief Destructor
     */
    ~ParticleState() {
        if (ptr_) {
            stonesoup_particle_state_free(ptr_);
        }
    }

    /**
     * @brief Move assignment
     */
    ParticleState& operator=(ParticleState&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                stonesoup_particle_state_free(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] std::size_t num_particles() const noexcept {
        return ptr_ ? ptr_->num_particles : 0;
    }

    /**
     * @brief Get particle state element
     */
    [[nodiscard]] double& particle_state(std::size_t particle, std::size_t i) {
        return ptr_->particles[particle].state_vector->data[i];
    }

    [[nodiscard]] const double& particle_state(std::size_t particle, std::size_t i) const {
        return ptr_->particles[particle].state_vector->data[i];
    }

    /**
     * @brief Get/set particle weight
     */
    [[nodiscard]] double& weight(std::size_t particle) {
        return ptr_->particles[particle].weight;
    }

    [[nodiscard]] const double& weight(std::size_t particle) const {
        return ptr_->particles[particle].weight;
    }

    /**
     * @brief Normalize all weights to sum to 1.0
     */
    void normalize_weights() {
        check_error(stonesoup_particle_state_normalize_weights(ptr_));
    }

    /**
     * @brief Get/set timestamp
     */
    [[nodiscard]] double timestamp() const noexcept { return ptr_->timestamp; }
    void set_timestamp(double t) noexcept { ptr_->timestamp = t; }

    /**
     * @brief Get underlying C pointer
     */
    [[nodiscard]] stonesoup_particle_state_t* get() noexcept { return ptr_; }
    [[nodiscard]] const stonesoup_particle_state_t* get() const noexcept { return ptr_; }

private:
    stonesoup_particle_state_t* ptr_ = nullptr;
};

/**
 * @brief Library initialization (RAII)
 */
class Library {
public:
    Library() {
        check_error(stonesoup_init());
    }

    ~Library() {
        stonesoup_cleanup();
    }

    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
};

} // namespace stonesoup

#endif // STONESOUP_HPP
