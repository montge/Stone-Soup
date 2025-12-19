/**
 * @file sci_stonesoup.c
 * @brief Scilab gateway functions for Stone Soup tracking framework
 *
 * This file provides the C gateway interface between Scilab and libstonesoup.
 * It implements Scilab-callable functions that wrap the core Stone Soup
 * tracking algorithms.
 *
 * Requires: Scilab 6.0+, libstonesoup
 */

#include <string.h>
#include <stdio.h>
#include "api_scilab.h"
#include "Scierror.h"
#include "localization.h"

#include "stonesoup/stonesoup.h"

/* Gateway function declarations */
int sci_stonesoup_version(scilabEnv env, int nin, int* in, int nout, int* out);
int sci_stonesoup_state_vector_create(scilabEnv env, int nin, int* in, int nout, int* out);
int sci_stonesoup_state_vector_add(scilabEnv env, int nin, int* in, int nout, int* out);
int sci_stonesoup_state_vector_norm(scilabEnv env, int nin, int* in, int nout, int* out);
int sci_stonesoup_covariance_identity(scilabEnv env, int nin, int* in, int nout, int* out);
int sci_stonesoup_kalman_predict(scilabEnv env, int nin, int* in, int nout, int* out);
int sci_stonesoup_kalman_update(scilabEnv env, int nin, int* in, int nout, int* out);

/* Helper function to convert Scilab error to libstonesoup error message */
static void stonesoup_scilab_error(stonesoup_error_t err, const char* func_name)
{
    const char* msg = stonesoup_error_string(err);
    Scierror(999, "%s: %s\n", func_name, msg);
}

/**
 * @brief Get Stone Soup version string
 *
 * Scilab usage: version = stonesoup_version()
 *
 * @return Version string
 */
int sci_stonesoup_version(scilabEnv env, int nin, int* in, int nout, int* out)
{
    const char* version = STONESOUP_VERSION;
    wchar_t* wversion;
    size_t len;

    /* Check arguments */
    if (nin != 0) {
        Scierror(77, "%s: Wrong number of input arguments: %d expected.\n",
                 "stonesoup_version", 0);
        return 1;
    }

    if (nout > 1) {
        Scierror(78, "%s: Wrong number of output arguments: %d expected.\n",
                 "stonesoup_version", 1);
        return 1;
    }

    /* Convert to wide string for Scilab */
    len = strlen(version) + 1;
    wversion = (wchar_t*)malloc(len * sizeof(wchar_t));
    if (!wversion) {
        Scierror(999, "%s: Memory allocation failed.\n", "stonesoup_version");
        return 1;
    }

    mbstowcs(wversion, version, len);

    /* Create output string */
    out[0] = scilab_createString(env, wversion);
    free(wversion);

    return 0;
}

/**
 * @brief Create a state vector (column vector)
 *
 * Scilab usage: sv = stonesoup_state_vector_create(dim)
 *               sv = stonesoup_state_vector_create(dim, value)
 *
 * @param dim Dimension of the state vector
 * @param value (optional) Fill value (default 0.0)
 * @return Column vector of specified dimension
 */
int sci_stonesoup_state_vector_create(scilabEnv env, int nin, int* in, int nout, int* out)
{
    double dim_d;
    double fill_value = 0.0;
    int dim;
    double* data;
    int i;

    /* Check arguments */
    if (nin < 1 || nin > 2) {
        Scierror(77, "%s: Wrong number of input arguments: %d or %d expected.\n",
                 "stonesoup_state_vector_create", 1, 2);
        return 1;
    }

    /* Get dimension */
    if (scilab_isDouble(env, in[0]) == 0 || scilab_isScalar(env, in[0]) == 0) {
        Scierror(999, "%s: Argument #%d must be a scalar.\n",
                 "stonesoup_state_vector_create", 1);
        return 1;
    }
    scilab_getDouble(env, in[0], &dim_d);
    dim = (int)dim_d;

    if (dim <= 0) {
        Scierror(999, "%s: Dimension must be positive.\n",
                 "stonesoup_state_vector_create");
        return 1;
    }

    /* Get optional fill value */
    if (nin == 2) {
        if (scilab_isDouble(env, in[1]) == 0 || scilab_isScalar(env, in[1]) == 0) {
            Scierror(999, "%s: Argument #%d must be a scalar.\n",
                     "stonesoup_state_vector_create", 2);
            return 1;
        }
        scilab_getDouble(env, in[1], &fill_value);
    }

    /* Create output matrix (column vector) */
    out[0] = scilab_createDoubleMatrix2d(env, dim, 1, 0);
    scilab_getDoubleArray(env, out[0], &data);

    /* Fill with value */
    for (i = 0; i < dim; i++) {
        data[i] = fill_value;
    }

    return 0;
}

/**
 * @brief Add two state vectors
 *
 * Scilab usage: result = stonesoup_state_vector_add(sv1, sv2)
 *
 * @param sv1 First state vector
 * @param sv2 Second state vector
 * @return Sum of the two vectors
 */
int sci_stonesoup_state_vector_add(scilabEnv env, int nin, int* in, int nout, int* out)
{
    double* data1;
    double* data2;
    double* result;
    int rows1, cols1, rows2, cols2;
    int i;

    /* Check arguments */
    if (nin != 2) {
        Scierror(77, "%s: Wrong number of input arguments: %d expected.\n",
                 "stonesoup_state_vector_add", 2);
        return 1;
    }

    /* Get first vector */
    if (scilab_isDouble(env, in[0]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_state_vector_add", 1);
        return 1;
    }
    scilab_getDoubleArray(env, in[0], &data1);
    scilab_getDim2d(env, in[0], &rows1, &cols1);

    /* Get second vector */
    if (scilab_isDouble(env, in[1]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_state_vector_add", 2);
        return 1;
    }
    scilab_getDoubleArray(env, in[1], &data2);
    scilab_getDim2d(env, in[1], &rows2, &cols2);

    /* Check dimensions match */
    if (rows1 != rows2 || cols1 != cols2) {
        Scierror(999, "%s: Arguments must have the same dimensions.\n",
                 "stonesoup_state_vector_add");
        return 1;
    }

    /* Create output */
    out[0] = scilab_createDoubleMatrix2d(env, rows1, cols1, 0);
    scilab_getDoubleArray(env, out[0], &result);

    /* Add vectors */
    for (i = 0; i < rows1 * cols1; i++) {
        result[i] = data1[i] + data2[i];
    }

    return 0;
}

/**
 * @brief Compute Euclidean norm of a state vector
 *
 * Scilab usage: n = stonesoup_state_vector_norm(sv)
 *
 * @param sv State vector
 * @return Euclidean norm
 */
int sci_stonesoup_state_vector_norm(scilabEnv env, int nin, int* in, int nout, int* out)
{
    double* data;
    int rows, cols;
    double norm = 0.0;
    int i, n;

    /* Check arguments */
    if (nin != 1) {
        Scierror(77, "%s: Wrong number of input arguments: %d expected.\n",
                 "stonesoup_state_vector_norm", 1);
        return 1;
    }

    /* Get vector */
    if (scilab_isDouble(env, in[0]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_state_vector_norm", 1);
        return 1;
    }
    scilab_getDoubleArray(env, in[0], &data);
    scilab_getDim2d(env, in[0], &rows, &cols);

    n = rows * cols;

    /* Compute norm */
    for (i = 0; i < n; i++) {
        norm += data[i] * data[i];
    }
    norm = sqrt(norm);

    /* Create output scalar */
    out[0] = scilab_createDouble(env, norm);

    return 0;
}

/**
 * @brief Create an identity covariance matrix
 *
 * Scilab usage: I = stonesoup_covariance_identity(dim)
 *
 * @param dim Dimension of the matrix
 * @return Identity matrix of specified dimension
 */
int sci_stonesoup_covariance_identity(scilabEnv env, int nin, int* in, int nout, int* out)
{
    double dim_d;
    int dim;
    double* data;
    int i, j;

    /* Check arguments */
    if (nin != 1) {
        Scierror(77, "%s: Wrong number of input arguments: %d expected.\n",
                 "stonesoup_covariance_identity", 1);
        return 1;
    }

    /* Get dimension */
    if (scilab_isDouble(env, in[0]) == 0 || scilab_isScalar(env, in[0]) == 0) {
        Scierror(999, "%s: Argument #%d must be a scalar.\n",
                 "stonesoup_covariance_identity", 1);
        return 1;
    }
    scilab_getDouble(env, in[0], &dim_d);
    dim = (int)dim_d;

    if (dim <= 0) {
        Scierror(999, "%s: Dimension must be positive.\n",
                 "stonesoup_covariance_identity");
        return 1;
    }

    /* Create output matrix */
    out[0] = scilab_createDoubleMatrix2d(env, dim, dim, 0);
    scilab_getDoubleArray(env, out[0], &data);

    /* Set to identity */
    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            data[i + j * dim] = (i == j) ? 1.0 : 0.0;
        }
    }

    return 0;
}

/**
 * @brief Kalman filter prediction step
 *
 * Scilab usage: [x_pred, P_pred] = stonesoup_kalman_predict(x, P, F, Q)
 *
 * @param x Prior state vector
 * @param P Prior covariance matrix
 * @param F State transition matrix
 * @param Q Process noise covariance
 * @return Predicted state and covariance
 */
int sci_stonesoup_kalman_predict(scilabEnv env, int nin, int* in, int nout, int* out)
{
    double* x_data;
    double* P_data;
    double* F_data;
    double* Q_data;
    double* x_pred_data;
    double* P_pred_data;
    int x_rows, x_cols;
    int P_rows, P_cols;
    int F_rows, F_cols;
    int Q_rows, Q_cols;
    int state_dim;

    stonesoup_gaussian_state_t* prior = NULL;
    stonesoup_gaussian_state_t* predicted = NULL;
    stonesoup_covariance_matrix_t* F_mat = NULL;
    stonesoup_covariance_matrix_t* Q_mat = NULL;
    stonesoup_error_t err;
    int i, j;

    /* Check arguments */
    if (nin != 4) {
        Scierror(77, "%s: Wrong number of input arguments: %d expected.\n",
                 "stonesoup_kalman_predict", 4);
        return 1;
    }

    if (nout < 1 || nout > 2) {
        Scierror(78, "%s: Wrong number of output arguments: %d to %d expected.\n",
                 "stonesoup_kalman_predict", 1, 2);
        return 1;
    }

    /* Get state vector */
    if (scilab_isDouble(env, in[0]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_predict", 1);
        return 1;
    }
    scilab_getDoubleArray(env, in[0], &x_data);
    scilab_getDim2d(env, in[0], &x_rows, &x_cols);
    state_dim = x_rows * x_cols;

    /* Get prior covariance */
    if (scilab_isDouble(env, in[1]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_predict", 2);
        return 1;
    }
    scilab_getDoubleArray(env, in[1], &P_data);
    scilab_getDim2d(env, in[1], &P_rows, &P_cols);

    /* Get transition matrix */
    if (scilab_isDouble(env, in[2]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_predict", 3);
        return 1;
    }
    scilab_getDoubleArray(env, in[2], &F_data);
    scilab_getDim2d(env, in[2], &F_rows, &F_cols);

    /* Get process noise */
    if (scilab_isDouble(env, in[3]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_predict", 4);
        return 1;
    }
    scilab_getDoubleArray(env, in[3], &Q_data);
    scilab_getDim2d(env, in[3], &Q_rows, &Q_cols);

    /* Validate dimensions */
    if (P_rows != state_dim || P_cols != state_dim) {
        Scierror(999, "%s: Covariance matrix dimensions must match state dimension.\n",
                 "stonesoup_kalman_predict");
        return 1;
    }
    if (F_rows != state_dim || F_cols != state_dim) {
        Scierror(999, "%s: Transition matrix dimensions must match state dimension.\n",
                 "stonesoup_kalman_predict");
        return 1;
    }
    if (Q_rows != state_dim || Q_cols != state_dim) {
        Scierror(999, "%s: Process noise dimensions must match state dimension.\n",
                 "stonesoup_kalman_predict");
        return 1;
    }

    /* Create libstonesoup structures */
    prior = stonesoup_gaussian_state_create(state_dim);
    predicted = stonesoup_gaussian_state_create(state_dim);
    F_mat = stonesoup_covariance_matrix_create(state_dim, state_dim);
    Q_mat = stonesoup_covariance_matrix_create(state_dim, state_dim);

    if (!prior || !predicted || !F_mat || !Q_mat) {
        Scierror(999, "%s: Memory allocation failed.\n", "stonesoup_kalman_predict");
        goto cleanup;
    }

    /* Copy data to libstonesoup structures */
    for (i = 0; i < state_dim; i++) {
        prior->state_vector->data[i] = x_data[i];
    }
    for (i = 0; i < state_dim; i++) {
        for (j = 0; j < state_dim; j++) {
            /* Scilab uses column-major, libstonesoup uses row-major */
            prior->covariance->data[i * state_dim + j] = P_data[i + j * state_dim];
            F_mat->data[i * state_dim + j] = F_data[i + j * state_dim];
            Q_mat->data[i * state_dim + j] = Q_data[i + j * state_dim];
        }
    }

    /* Call libstonesoup */
    err = stonesoup_kalman_predict(prior, F_mat, Q_mat, predicted);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_scilab_error(err, "stonesoup_kalman_predict");
        goto cleanup;
    }

    /* Create output state vector */
    out[0] = scilab_createDoubleMatrix2d(env, state_dim, 1, 0);
    scilab_getDoubleArray(env, out[0], &x_pred_data);
    for (i = 0; i < state_dim; i++) {
        x_pred_data[i] = predicted->state_vector->data[i];
    }

    /* Create output covariance if requested */
    if (nout >= 2) {
        out[1] = scilab_createDoubleMatrix2d(env, state_dim, state_dim, 0);
        scilab_getDoubleArray(env, out[1], &P_pred_data);
        for (i = 0; i < state_dim; i++) {
            for (j = 0; j < state_dim; j++) {
                /* Convert row-major back to column-major */
                P_pred_data[i + j * state_dim] = predicted->covariance->data[i * state_dim + j];
            }
        }
    }

cleanup:
    if (prior) stonesoup_gaussian_state_free(prior);
    if (predicted) stonesoup_gaussian_state_free(predicted);
    if (F_mat) stonesoup_covariance_matrix_free(F_mat);
    if (Q_mat) stonesoup_covariance_matrix_free(Q_mat);

    return (err != STONESOUP_SUCCESS) ? 1 : 0;
}

/**
 * @brief Kalman filter update step
 *
 * Scilab usage: [x_post, P_post] = stonesoup_kalman_update(x_pred, P_pred, z, H, R)
 *
 * @param x_pred Predicted state vector
 * @param P_pred Predicted covariance matrix
 * @param z Measurement vector
 * @param H Measurement matrix
 * @param R Measurement noise covariance
 * @return Posterior state and covariance
 */
int sci_stonesoup_kalman_update(scilabEnv env, int nin, int* in, int nout, int* out)
{
    double* x_pred_data;
    double* P_pred_data;
    double* z_data;
    double* H_data;
    double* R_data;
    double* x_post_data;
    double* P_post_data;
    int x_rows, x_cols;
    int P_rows, P_cols;
    int z_rows, z_cols;
    int H_rows, H_cols;
    int R_rows, R_cols;
    int state_dim, meas_dim;

    stonesoup_gaussian_state_t* predicted = NULL;
    stonesoup_gaussian_state_t* posterior = NULL;
    stonesoup_state_vector_t* measurement = NULL;
    stonesoup_covariance_matrix_t* H_mat = NULL;
    stonesoup_covariance_matrix_t* R_mat = NULL;
    stonesoup_error_t err = STONESOUP_SUCCESS;
    int i, j;

    /* Check arguments */
    if (nin != 5) {
        Scierror(77, "%s: Wrong number of input arguments: %d expected.\n",
                 "stonesoup_kalman_update", 5);
        return 1;
    }

    if (nout < 1 || nout > 2) {
        Scierror(78, "%s: Wrong number of output arguments: %d to %d expected.\n",
                 "stonesoup_kalman_update", 1, 2);
        return 1;
    }

    /* Get predicted state vector */
    if (scilab_isDouble(env, in[0]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_update", 1);
        return 1;
    }
    scilab_getDoubleArray(env, in[0], &x_pred_data);
    scilab_getDim2d(env, in[0], &x_rows, &x_cols);
    state_dim = x_rows * x_cols;

    /* Get predicted covariance */
    if (scilab_isDouble(env, in[1]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_update", 2);
        return 1;
    }
    scilab_getDoubleArray(env, in[1], &P_pred_data);
    scilab_getDim2d(env, in[1], &P_rows, &P_cols);

    /* Get measurement */
    if (scilab_isDouble(env, in[2]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_update", 3);
        return 1;
    }
    scilab_getDoubleArray(env, in[2], &z_data);
    scilab_getDim2d(env, in[2], &z_rows, &z_cols);
    meas_dim = z_rows * z_cols;

    /* Get measurement matrix */
    if (scilab_isDouble(env, in[3]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_update", 4);
        return 1;
    }
    scilab_getDoubleArray(env, in[3], &H_data);
    scilab_getDim2d(env, in[3], &H_rows, &H_cols);

    /* Get measurement noise */
    if (scilab_isDouble(env, in[4]) == 0) {
        Scierror(999, "%s: Argument #%d must be a real matrix.\n",
                 "stonesoup_kalman_update", 5);
        return 1;
    }
    scilab_getDoubleArray(env, in[4], &R_data);
    scilab_getDim2d(env, in[4], &R_rows, &R_cols);

    /* Validate dimensions */
    if (P_rows != state_dim || P_cols != state_dim) {
        Scierror(999, "%s: Covariance matrix dimensions must match state dimension.\n",
                 "stonesoup_kalman_update");
        return 1;
    }
    if (H_rows != meas_dim || H_cols != state_dim) {
        Scierror(999, "%s: Measurement matrix must be meas_dim x state_dim.\n",
                 "stonesoup_kalman_update");
        return 1;
    }
    if (R_rows != meas_dim || R_cols != meas_dim) {
        Scierror(999, "%s: Measurement noise must be meas_dim x meas_dim.\n",
                 "stonesoup_kalman_update");
        return 1;
    }

    /* Create libstonesoup structures */
    predicted = stonesoup_gaussian_state_create(state_dim);
    posterior = stonesoup_gaussian_state_create(state_dim);
    measurement = stonesoup_state_vector_create(meas_dim);
    H_mat = stonesoup_covariance_matrix_create(meas_dim, state_dim);
    R_mat = stonesoup_covariance_matrix_create(meas_dim, meas_dim);

    if (!predicted || !posterior || !measurement || !H_mat || !R_mat) {
        Scierror(999, "%s: Memory allocation failed.\n", "stonesoup_kalman_update");
        err = STONESOUP_ERROR_ALLOCATION;
        goto cleanup;
    }

    /* Copy data to libstonesoup structures */
    for (i = 0; i < state_dim; i++) {
        predicted->state_vector->data[i] = x_pred_data[i];
    }
    for (i = 0; i < state_dim; i++) {
        for (j = 0; j < state_dim; j++) {
            predicted->covariance->data[i * state_dim + j] = P_pred_data[i + j * state_dim];
        }
    }
    for (i = 0; i < meas_dim; i++) {
        measurement->data[i] = z_data[i];
    }
    for (i = 0; i < meas_dim; i++) {
        for (j = 0; j < state_dim; j++) {
            H_mat->data[i * state_dim + j] = H_data[i + j * meas_dim];
        }
    }
    for (i = 0; i < meas_dim; i++) {
        for (j = 0; j < meas_dim; j++) {
            R_mat->data[i * meas_dim + j] = R_data[i + j * meas_dim];
        }
    }

    /* Call libstonesoup */
    err = stonesoup_kalman_update(predicted, measurement, H_mat, R_mat, posterior);
    if (err != STONESOUP_SUCCESS) {
        stonesoup_scilab_error(err, "stonesoup_kalman_update");
        goto cleanup;
    }

    /* Create output state vector */
    out[0] = scilab_createDoubleMatrix2d(env, state_dim, 1, 0);
    scilab_getDoubleArray(env, out[0], &x_post_data);
    for (i = 0; i < state_dim; i++) {
        x_post_data[i] = posterior->state_vector->data[i];
    }

    /* Create output covariance if requested */
    if (nout >= 2) {
        out[1] = scilab_createDoubleMatrix2d(env, state_dim, state_dim, 0);
        scilab_getDoubleArray(env, out[1], &P_post_data);
        for (i = 0; i < state_dim; i++) {
            for (j = 0; j < state_dim; j++) {
                P_post_data[i + j * state_dim] = posterior->covariance->data[i * state_dim + j];
            }
        }
    }

cleanup:
    if (predicted) stonesoup_gaussian_state_free(predicted);
    if (posterior) stonesoup_gaussian_state_free(posterior);
    if (measurement) stonesoup_state_vector_free(measurement);
    if (H_mat) stonesoup_covariance_matrix_free(H_mat);
    if (R_mat) stonesoup_covariance_matrix_free(R_mat);

    return (err != STONESOUP_SUCCESS) ? 1 : 0;
}
