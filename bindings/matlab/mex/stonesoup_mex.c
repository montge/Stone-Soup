/**
 * @file stonesoup_mex.c
 * @brief MATLAB MEX gateway for Stone Soup tracking framework
 *
 * This file provides MEX interface functions for calling libstonesoup
 * from MATLAB and GNU Octave.
 *
 * Compile with:
 *   mex -I../../libstonesoup/include -L../../libstonesoup/build -lstonesoup stonesoup_mex.c
 *
 * For Octave:
 *   mkoctfile --mex -I../../libstonesoup/include -L../../libstonesoup/build -lstonesoup stonesoup_mex.c
 */

#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <math.h>

#include "stonesoup/stonesoup.h"

/* Command dispatcher */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char cmd[64];

    /* Check for command string */
    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("stonesoup:invalidInput",
            "First argument must be a command string.\n"
            "Available commands: version, kalman_predict, kalman_update");
        return;
    }

    /* Get command */
    mxGetString(prhs[0], cmd, sizeof(cmd));

    /* Dispatch commands */
    if (strcmp(cmd, "version") == 0) {
        /* Return version string */
        plhs[0] = mxCreateString(STONESOUP_VERSION);
    }
    else if (strcmp(cmd, "kalman_predict") == 0) {
        /* Kalman prediction: [x_pred, P_pred] = stonesoup_mex('kalman_predict', x, P, F, Q) */
        double *x, *P, *F, *Q;
        double *x_pred, *P_pred;
        mwSize state_dim;
        size_t i, j;
        stonesoup_gaussian_state_t *prior = NULL;
        stonesoup_gaussian_state_t *predicted = NULL;
        stonesoup_covariance_matrix_t *F_mat = NULL;
        stonesoup_covariance_matrix_t *Q_mat = NULL;
        stonesoup_error_t err;

        if (nrhs != 5) {
            mexErrMsgIdAndTxt("stonesoup:invalidInput",
                "kalman_predict requires 4 arguments: x, P, F, Q");
            return;
        }

        /* Get input dimensions */
        state_dim = mxGetNumberOfElements(prhs[1]);

        /* Validate inputs */
        if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) ||
            !mxIsDouble(prhs[3]) || !mxIsDouble(prhs[4])) {
            mexErrMsgIdAndTxt("stonesoup:invalidInput",
                "All inputs must be double arrays");
            return;
        }

        /* Get input data pointers */
        x = mxGetPr(prhs[1]);
        P = mxGetPr(prhs[2]);
        F = mxGetPr(prhs[3]);
        Q = mxGetPr(prhs[4]);

        /* Create libstonesoup structures */
        prior = stonesoup_gaussian_state_create(state_dim);
        predicted = stonesoup_gaussian_state_create(state_dim);
        F_mat = stonesoup_covariance_matrix_create(state_dim, state_dim);
        Q_mat = stonesoup_covariance_matrix_create(state_dim, state_dim);

        if (!prior || !predicted || !F_mat || !Q_mat) {
            mexErrMsgIdAndTxt("stonesoup:memoryError",
                "Failed to allocate memory");
            goto kalman_predict_cleanup;
        }

        /* Copy data (MATLAB is column-major, libstonesoup is row-major) */
        for (i = 0; i < state_dim; i++) {
            prior->state_vector->data[i] = x[i];
        }
        for (i = 0; i < state_dim; i++) {
            for (j = 0; j < state_dim; j++) {
                /* MATLAB: P(i,j) = P[i + j*state_dim] (column-major) */
                /* libstonesoup: P[i*cols + j] (row-major) */
                prior->covariance->data[i * state_dim + j] = P[i + j * state_dim];
                F_mat->data[i * state_dim + j] = F[i + j * state_dim];
                Q_mat->data[i * state_dim + j] = Q[i + j * state_dim];
            }
        }

        /* Call libstonesoup */
        err = stonesoup_kalman_predict(prior, F_mat, Q_mat, predicted);
        if (err != STONESOUP_SUCCESS) {
            mexErrMsgIdAndTxt("stonesoup:kalmanError",
                "Kalman predict failed: %s", stonesoup_error_string(err));
            goto kalman_predict_cleanup;
        }

        /* Create outputs */
        plhs[0] = mxCreateDoubleMatrix(state_dim, 1, mxREAL);
        x_pred = mxGetPr(plhs[0]);
        for (i = 0; i < state_dim; i++) {
            x_pred[i] = predicted->state_vector->data[i];
        }

        if (nlhs >= 2) {
            plhs[1] = mxCreateDoubleMatrix(state_dim, state_dim, mxREAL);
            P_pred = mxGetPr(plhs[1]);
            for (i = 0; i < state_dim; i++) {
                for (j = 0; j < state_dim; j++) {
                    P_pred[i + j * state_dim] = predicted->covariance->data[i * state_dim + j];
                }
            }
        }

kalman_predict_cleanup:
        if (prior) stonesoup_gaussian_state_free(prior);
        if (predicted) stonesoup_gaussian_state_free(predicted);
        if (F_mat) stonesoup_covariance_matrix_free(F_mat);
        if (Q_mat) stonesoup_covariance_matrix_free(Q_mat);
    }
    else if (strcmp(cmd, "kalman_update") == 0) {
        /* Kalman update: [x_post, P_post] = stonesoup_mex('kalman_update', x, P, z, H, R) */
        double *x, *P, *z, *H, *R;
        double *x_post, *P_post;
        mwSize state_dim, meas_dim;
        size_t i, j;
        stonesoup_gaussian_state_t *predicted = NULL;
        stonesoup_gaussian_state_t *posterior = NULL;
        stonesoup_state_vector_t *measurement = NULL;
        stonesoup_covariance_matrix_t *H_mat = NULL;
        stonesoup_covariance_matrix_t *R_mat = NULL;
        stonesoup_error_t err;

        if (nrhs != 6) {
            mexErrMsgIdAndTxt("stonesoup:invalidInput",
                "kalman_update requires 5 arguments: x, P, z, H, R");
            return;
        }

        /* Get input dimensions */
        state_dim = mxGetNumberOfElements(prhs[1]);
        meas_dim = mxGetNumberOfElements(prhs[3]);

        /* Validate inputs */
        if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) ||
            !mxIsDouble(prhs[3]) || !mxIsDouble(prhs[4]) ||
            !mxIsDouble(prhs[5])) {
            mexErrMsgIdAndTxt("stonesoup:invalidInput",
                "All inputs must be double arrays");
            return;
        }

        /* Get input data pointers */
        x = mxGetPr(prhs[1]);
        P = mxGetPr(prhs[2]);
        z = mxGetPr(prhs[3]);
        H = mxGetPr(prhs[4]);
        R = mxGetPr(prhs[5]);

        /* Create libstonesoup structures */
        predicted = stonesoup_gaussian_state_create(state_dim);
        posterior = stonesoup_gaussian_state_create(state_dim);
        measurement = stonesoup_state_vector_create(meas_dim);
        H_mat = stonesoup_covariance_matrix_create(meas_dim, state_dim);
        R_mat = stonesoup_covariance_matrix_create(meas_dim, meas_dim);

        if (!predicted || !posterior || !measurement || !H_mat || !R_mat) {
            mexErrMsgIdAndTxt("stonesoup:memoryError",
                "Failed to allocate memory");
            goto kalman_update_cleanup;
        }

        /* Copy data */
        for (i = 0; i < state_dim; i++) {
            predicted->state_vector->data[i] = x[i];
        }
        for (i = 0; i < state_dim; i++) {
            for (j = 0; j < state_dim; j++) {
                predicted->covariance->data[i * state_dim + j] = P[i + j * state_dim];
            }
        }
        for (i = 0; i < meas_dim; i++) {
            measurement->data[i] = z[i];
        }
        for (i = 0; i < meas_dim; i++) {
            for (j = 0; j < state_dim; j++) {
                H_mat->data[i * state_dim + j] = H[i + j * meas_dim];
            }
        }
        for (i = 0; i < meas_dim; i++) {
            for (j = 0; j < meas_dim; j++) {
                R_mat->data[i * meas_dim + j] = R[i + j * meas_dim];
            }
        }

        /* Call libstonesoup */
        err = stonesoup_kalman_update(predicted, measurement, H_mat, R_mat, posterior);
        if (err != STONESOUP_SUCCESS) {
            mexErrMsgIdAndTxt("stonesoup:kalmanError",
                "Kalman update failed: %s", stonesoup_error_string(err));
            goto kalman_update_cleanup;
        }

        /* Create outputs */
        plhs[0] = mxCreateDoubleMatrix(state_dim, 1, mxREAL);
        x_post = mxGetPr(plhs[0]);
        for (i = 0; i < state_dim; i++) {
            x_post[i] = posterior->state_vector->data[i];
        }

        if (nlhs >= 2) {
            plhs[1] = mxCreateDoubleMatrix(state_dim, state_dim, mxREAL);
            P_post = mxGetPr(plhs[1]);
            for (i = 0; i < state_dim; i++) {
                for (j = 0; j < state_dim; j++) {
                    P_post[i + j * state_dim] = posterior->covariance->data[i * state_dim + j];
                }
            }
        }

kalman_update_cleanup:
        if (predicted) stonesoup_gaussian_state_free(predicted);
        if (posterior) stonesoup_gaussian_state_free(posterior);
        if (measurement) stonesoup_state_vector_free(measurement);
        if (H_mat) stonesoup_covariance_matrix_free(H_mat);
        if (R_mat) stonesoup_covariance_matrix_free(R_mat);
    }
    else {
        mexErrMsgIdAndTxt("stonesoup:unknownCommand",
            "Unknown command: %s\n"
            "Available commands: version, kalman_predict, kalman_update", cmd);
    }
}
