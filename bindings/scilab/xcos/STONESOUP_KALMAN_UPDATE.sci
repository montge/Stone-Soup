// STONESOUP_KALMAN_UPDATE Xcos block for Kalman filter update
//
// This Xcos block implements the Kalman filter update (correction) step.
//
// Inputs:
//   Port 1: Predicted state vector (state_dim x 1)
//   Port 2: Predicted covariance matrix as vector (state_dim^2 x 1)
//   Port 3: Measurement vector (meas_dim x 1)
//
// Outputs:
//   Port 1: Posterior state vector (state_dim x 1)
//   Port 2: Posterior covariance matrix as vector (state_dim^2 x 1)
//   Port 3: Innovation vector (meas_dim x 1)
//
// Parameters:
//   state_dim: State vector dimension
//   meas_dim: Measurement vector dimension
//   H: Measurement matrix (flattened)
//   R: Measurement noise covariance (flattened)
//
// See also: STONESOUP_KALMAN_PREDICT

function [x, y, typ] = STONESOUP_KALMAN_UPDATE(job, arg1, arg2)
    x = [];
    y = [];
    typ = [];

    select job
    case "set" then
        x = arg1;
        graphics = arg1.graphics;
        exprs = graphics.exprs;

        while %t do
            [ok, state_dim, meas_dim, H_str, R_str, exprs] = scicos_getvalue( ..
                "Set Kalman Updater Parameters", ..
                ["State dimension (integer)"; ..
                 "Measurement dimension (integer)"; ..
                 "H: Measurement matrix (flattened row-major)"; ..
                 "R: Measurement noise covariance (flattened row-major)"], ..
                list("vec", 1, "vec", 1, "vec", -1, "vec", -1), ..
                exprs);

            if ~ok then
                break;
            end

            state_dim = int(state_dim);
            meas_dim = int(meas_dim);

            if state_dim < 1 | meas_dim < 1 then
                message("Dimensions must be at least 1");
                ok = %f;
            end

            H = matrix(evstr(H_str), meas_dim, state_dim);
            R = matrix(evstr(R_str), meas_dim, meas_dim);

            if ok then
                graphics.exprs = exprs;
                model = arg1.model;
                model.rpar = [state_dim; meas_dim; H(:); R(:)];
                model.in = [state_dim; state_dim * state_dim; meas_dim];
                model.out = [state_dim; state_dim * state_dim; meas_dim];
                x.graphics = graphics;
                x.model = model;
                break;
            end
        end

    case "define" then
        // Default: position-only measurement of 4-state model
        state_dim = 4;
        meas_dim = 2;
        H = [1, 0, 0, 0; ..
             0, 0, 1, 0];
        R = 0.5 * eye(2, 2);

        model = scicos_model();
        model.sim = list("stonesoup_kalman_update_sim", 4);
        model.in = [state_dim; state_dim * state_dim; meas_dim];
        model.out = [state_dim; state_dim * state_dim; meas_dim];
        model.rpar = [state_dim; meas_dim; H(:); R(:)];
        model.blocktype = "c";
        model.dep_ut = [%t, %f];

        exprs = [string(state_dim); ..
                 string(meas_dim); ..
                 sci2exp(H(:)'); ..
                 sci2exp(R(:)')];

        gr_i = ["xstringb(orig(1), orig(2), ""Kalman"", sz(1), sz(2))"; ..
                "xstringb(orig(1), orig(2)+sz(2)/3, ""Update"", sz(1), sz(2)/3)"];

        x = standard_define([3, 2], model, exprs, gr_i);

    case "details" then
        x = ["STONESOUP_KALMAN_UPDATE"; ..
             "Kalman Filter Update"; ..
             "Implements measurement update with Joseph form covariance"];

    else
        break;
    end
endfunction

// Simulation function
function block = stonesoup_kalman_update_sim(block, flag)
    select flag
    case 1 then
        // Output computation
        rpar = block.rpar;
        state_dim = int(rpar(1));
        meas_dim = int(rpar(2));
        n_state2 = state_dim * state_dim;
        n_meas2 = meas_dim * meas_dim;

        // Extract H and R from rpar
        H = matrix(rpar(3:2+meas_dim*state_dim), meas_dim, state_dim);
        R = matrix(rpar(3+meas_dim*state_dim:2+meas_dim*state_dim+n_meas2), meas_dim, meas_dim);

        // Get inputs
        x_pred = block.inptr(1);
        P_pred_vec = block.inptr(2);
        z = block.inptr(3);

        P_pred = matrix(P_pred_vec, state_dim, state_dim);

        // Kalman update equations
        // Innovation
        y = z - H * x_pred;

        // Innovation covariance
        S = H * P_pred * H' + R;

        // Kalman gain
        K = P_pred * H' * inv(S);

        // Posterior state
        x_post = x_pred + K * y;

        // Posterior covariance (Joseph form for numerical stability)
        I_KH = eye(state_dim, state_dim) - K * H;
        P_post = I_KH * P_pred * I_KH' + K * R * K';

        // Set outputs
        block.outptr(1) = x_post;
        block.outptr(2) = P_post(:);
        block.outptr(3) = y;

    case 4 then
        // Initialization
        block.outptr(1) = zeros(block.outsz(1), 1);
        block.outptr(2) = zeros(block.outsz(2), 1);
        block.outptr(3) = zeros(block.outsz(3), 1);

    case 5 then
        // Termination - nothing to do

    end
endfunction
