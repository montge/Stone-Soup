// STONESOUP_KALMAN_PREDICT Xcos block for Kalman filter prediction
//
// This Xcos block implements the Kalman filter prediction step.
//
// Inputs:
//   Port 1: State vector (state_dim x 1)
//   Port 2: Covariance matrix (state_dim x state_dim) as vector
//
// Outputs:
//   Port 1: Predicted state vector (state_dim x 1)
//   Port 2: Predicted covariance matrix as vector
//
// Parameters:
//   state_dim: State vector dimension
//   F: State transition matrix (flattened)
//   Q: Process noise covariance (flattened)
//
// See also: STONESOUP_KALMAN_UPDATE

function [x, y, typ] = STONESOUP_KALMAN_PREDICT(job, arg1, arg2)
    x = [];
    y = [];
    typ = [];

    select job
    case "set" then
        x = arg1;
        graphics = arg1.graphics;
        exprs = graphics.exprs;

        while %t do
            [ok, state_dim, F_str, Q_str, exprs] = scicos_getvalue( ..
                "Set Kalman Predictor Parameters", ..
                ["State dimension (integer)"; ..
                 "F: Transition matrix (flattened row-major)"; ..
                 "Q: Process noise covariance (flattened row-major)"], ..
                list("vec", 1, "vec", -1, "vec", -1), ..
                exprs);

            if ~ok then
                break;
            end

            state_dim = int(state_dim);
            if state_dim < 1 then
                message("State dimension must be at least 1");
                ok = %f;
            end

            F = matrix(evstr(F_str), state_dim, state_dim);
            Q = matrix(evstr(Q_str), state_dim, state_dim);

            if ok then
                graphics.exprs = exprs;
                model = arg1.model;
                model.rpar = [state_dim; F(:); Q(:)];
                model.in = [state_dim; state_dim * state_dim];
                model.out = [state_dim; state_dim * state_dim];
                x.graphics = graphics;
                x.model = model;
                break;
            end
        end

    case "define" then
        // Default: 4-state constant velocity model
        state_dim = 4;
        dt = 1;
        F = [1, dt, 0, 0; ..
             0, 1, 0, 0; ..
             0, 0, 1, dt; ..
             0, 0, 0, 1];
        Q = 0.1 * eye(4, 4);

        model = scicos_model();
        model.sim = list("stonesoup_kalman_predict_sim", 4);
        model.in = [state_dim; state_dim * state_dim];
        model.out = [state_dim; state_dim * state_dim];
        model.rpar = [state_dim; F(:); Q(:)];
        model.blocktype = "c";
        model.dep_ut = [%t, %f];

        exprs = [string(state_dim); ..
                 sci2exp(F(:)'); ..
                 sci2exp(Q(:)')];

        gr_i = ["xstringb(orig(1), orig(2), ""Kalman"", sz(1), sz(2))"; ..
                "xstringb(orig(1), orig(2)+sz(2)/3, ""Predict"", sz(1), sz(2)/3)"];

        x = standard_define([3, 2], model, exprs, gr_i);

    case "details" then
        // Return block details
        x = ["STONESOUP_KALMAN_PREDICT"; ..
             "Kalman Filter Prediction"; ..
             "Implements state prediction: x_pred = F*x, P_pred = F*P*F'' + Q"];

    else
        break;
    end
endfunction

// Simulation function
function block = stonesoup_kalman_predict_sim(block, flag)
    select flag
    case 1 then
        // Output computation
        rpar = block.rpar;
        state_dim = int(rpar(1));
        n2 = state_dim * state_dim;

        // Extract F and Q from rpar
        F = matrix(rpar(2:1+n2), state_dim, state_dim);
        Q = matrix(rpar(2+n2:1+2*n2), state_dim, state_dim);

        // Get inputs
        x = block.inptr(1);
        P_vec = block.inptr(2);
        P = matrix(P_vec, state_dim, state_dim);

        // Kalman prediction
        x_pred = F * x;
        P_pred = F * P * F' + Q;

        // Set outputs
        block.outptr(1) = x_pred;
        block.outptr(2) = P_pred(:);

    case 4 then
        // Initialization
        block.outptr(1) = zeros(block.outsz(1), 1);
        block.outptr(2) = zeros(block.outsz(2), 1);

    case 5 then
        // Termination - nothing to do

    end
endfunction
