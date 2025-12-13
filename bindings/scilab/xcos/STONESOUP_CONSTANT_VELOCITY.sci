// STONESOUP_CONSTANT_VELOCITY Xcos block for constant velocity motion model
//
// This Xcos block generates transition matrix F and process noise Q
// for a constant velocity motion model.
//
// Inputs:
//   Port 1: Time step dt (scalar)
//
// Outputs:
//   Port 1: Transition matrix F (state_dim^2 x 1, vectorized)
//   Port 2: Process noise Q (state_dim^2 x 1, vectorized)
//
// Parameters:
//   spatial_dims: Number of spatial dimensions (1, 2, or 3)
//   q: Process noise intensity (acceleration variance)
//
// Description:
//   State vector format: [x, vx, y, vy, z, vz] for 3D
//   Transition: position += velocity * dt
//
// See also: STONESOUP_KALMAN_PREDICT

function [x, y, typ] = STONESOUP_CONSTANT_VELOCITY(job, arg1, arg2)
    x = [];
    y = [];
    typ = [];

    select job
    case "set" then
        x = arg1;
        graphics = arg1.graphics;
        exprs = graphics.exprs;

        while %t do
            [ok, spatial_dims, q, exprs] = scicos_getvalue( ..
                "Set Constant Velocity Model Parameters", ..
                ["Spatial dimensions (1, 2, or 3)"; ..
                 "Process noise intensity q (acceleration variance)"], ..
                list("vec", 1, "vec", 1), ..
                exprs);

            if ~ok then
                break;
            end

            spatial_dims = int(spatial_dims);
            if spatial_dims < 1 | spatial_dims > 3 then
                message("Spatial dimensions must be 1, 2, or 3");
                ok = %f;
            end

            if ok then
                state_dim = spatial_dims * 2;
                graphics.exprs = exprs;
                model = arg1.model;
                model.rpar = [spatial_dims; q];
                model.out = [state_dim * state_dim; state_dim * state_dim];
                x.graphics = graphics;
                x.model = model;
                break;
            end
        end

    case "define" then
        // Default: 2D tracking
        spatial_dims = 2;
        q = 0.1;
        state_dim = spatial_dims * 2;

        model = scicos_model();
        model.sim = list("stonesoup_constant_velocity_sim", 4);
        model.in = 1;  // dt input
        model.out = [state_dim * state_dim; state_dim * state_dim];
        model.rpar = [spatial_dims; q];
        model.blocktype = "c";
        model.dep_ut = [%t, %f];

        exprs = [string(spatial_dims); string(q)];

        gr_i = ["xstringb(orig(1), orig(2), ""Const"", sz(1), sz(2))"; ..
                "xstringb(orig(1), orig(2)+sz(2)/3, ""Velocity"", sz(1), sz(2)/3)"];

        x = standard_define([3, 2], model, exprs, gr_i);

    case "details" then
        x = ["STONESOUP_CONSTANT_VELOCITY"; ..
             "Constant Velocity Motion Model"; ..
             "Generates F and Q matrices for constant velocity tracking"];

    else
        break;
    end
endfunction

// Simulation function
function block = stonesoup_constant_velocity_sim(block, flag)
    select flag
    case 1 then
        // Output computation
        rpar = block.rpar;
        spatial_dims = int(rpar(1));
        q = rpar(2);
        state_dim = spatial_dims * 2;

        // Get time step
        dt = block.inptr(1);

        // Build transition matrix F
        // For each spatial dimension: [pos; vel] -> [pos + vel*dt; vel]
        F = eye(state_dim, state_dim);
        for i = 1:spatial_dims
            pos_idx = (i-1)*2 + 1;
            vel_idx = pos_idx + 1;
            F(pos_idx, vel_idx) = dt;
        end

        // Build process noise Q (discrete white noise acceleration)
        // For each dimension: Q_block = q * [dt^4/4, dt^3/2; dt^3/2, dt^2]
        Q = zeros(state_dim, state_dim);
        Q_block = q * [dt^4/4, dt^3/2; dt^3/2, dt^2];
        for i = 1:spatial_dims
            idx = (i-1)*2 + (1:2);
            Q(idx, idx) = Q_block;
        end

        // Set outputs (vectorized column-major for Scilab)
        block.outptr(1) = F(:);
        block.outptr(2) = Q(:);

    case 4 then
        // Initialization
        spatial_dims = int(block.rpar(1));
        state_dim = spatial_dims * 2;
        n2 = state_dim * state_dim;
        block.outptr(1) = eye(state_dim, state_dim);
        block.outptr(1) = block.outptr(1)(:);
        block.outptr(2) = zeros(n2, 1);

    case 5 then
        // Termination - nothing to do

    end
endfunction
