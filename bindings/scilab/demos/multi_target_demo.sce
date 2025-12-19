// Stone Soup Multi-Target Tracking Demo
// Demonstrates tracking multiple targets simultaneously using Kalman filters
//
// This demo simulates 3 targets moving with constant velocity and
// shows how to maintain and update multiple track states.

mode(-1);
lines(0);

// Load Stone Soup
demo_path = get_absolute_file_path("multi_target_demo.sce");
exec(demo_path + "../loader.sce", -1);

mprintf("===========================================\n");
mprintf("Stone Soup Multi-Target Tracking Demo\n");
mprintf("===========================================\n\n");

// Simulation parameters
dt = 1.0;           // Time step (seconds)
num_steps = 20;     // Number of time steps
num_targets = 3;    // Number of targets to track

// Motion model: Constant Velocity in 2D
// State: [x, vx, y, vy]
F = [1, dt, 0, 0;
     0, 1,  0, 0;
     0, 0,  1, dt;
     0, 0,  0, 1];

// Process noise
q = 0.5;  // Acceleration noise standard deviation
Q = q^2 * [dt^3/3, dt^2/2, 0,      0;
           dt^2/2, dt,     0,      0;
           0,      0,      dt^3/3, dt^2/2;
           0,      0,      dt^2/2, dt];

// Measurement model: Position only
H = [1, 0, 0, 0;
     0, 0, 1, 0];

// Measurement noise
r = 2.0;  // Position measurement noise (meters)
R = r^2 * eye(2, 2);

// Initialize targets with different initial conditions
mprintf("Initializing %d targets...\n\n", num_targets);

// Target 1: Moving right
target1_truth = [0; 5; 0; 0];      // Start at origin, moving right at 5 m/s
target1_state = GaussianState([0; 0; 0; 0], 100*eye(4,4));

// Target 2: Moving diagonally
target2_truth = [50; -3; 50; 2];   // Start at (50,50), moving left and up
target2_state = GaussianState([50; 0; 50; 0], 100*eye(4,4));

// Target 3: Moving in a curve (we'll add turn noise)
target3_truth = [25; 4; 0; 4];     // Start at (25,0), moving diagonally
target3_state = GaussianState([25; 0; 0; 0], 100*eye(4,4));

// Store all targets
targets_truth = list(target1_truth, target2_truth, target3_truth);
targets_state = list(target1_state, target2_state, target3_state);

// Storage for plotting
history_truth = zeros(num_targets, 2, num_steps);
history_est = zeros(num_targets, 2, num_steps);
history_meas = zeros(num_targets, 2, num_steps);

// Main tracking loop
mprintf("Running simulation for %d time steps...\n\n", num_steps);

for t = 1:num_steps
    mprintf("--- Time step %d ---\n", t);

    for i = 1:num_targets
        // Get current truth and state
        truth = targets_truth(i);
        state = targets_state(i);

        // Propagate truth (with small random acceleration)
        truth = F * truth + [0; 0.1*rand(1,"normal"); 0; 0.1*rand(1,"normal")];
        targets_truth(i) = truth;

        // Generate noisy measurement
        z = H * truth + r * rand(2, 1, "normal");

        // Kalman prediction
        [x_pred, P_pred] = kalman_predict(state.mean, state.covar, F, Q);

        // Kalman update
        [x_upd, P_upd] = kalman_update(x_pred, P_pred, z, H, R);

        // Store updated state
        state.mean = x_upd;
        state.covar = P_upd;
        targets_state(i) = state;

        // Store for plotting
        history_truth(i, :, t) = [truth(1), truth(3)];
        history_est(i, :, t) = [x_upd(1), x_upd(3)];
        history_meas(i, :, t) = z';

        // Display
        mprintf("Target %d: True=[%.1f, %.1f], Est=[%.1f, %.1f], Err=%.2f\n", ...
            i, truth(1), truth(3), x_upd(1), x_upd(3), ...
            norm([truth(1)-x_upd(1); truth(3)-x_upd(3)]));
    end
    mprintf("\n");
end

// Calculate final statistics
mprintf("===========================================\n");
mprintf("Final Results\n");
mprintf("===========================================\n\n");

for i = 1:num_targets
    truth = targets_truth(i);
    state = targets_state(i);

    pos_error = norm([truth(1)-state.mean(1); truth(3)-state.mean(3)]);
    vel_error = norm([truth(2)-state.mean(2); truth(4)-state.mean(4)]);

    mprintf("Target %d:\n", i);
    mprintf("  Final position error: %.2f m\n", pos_error);
    mprintf("  Final velocity error: %.2f m/s\n", vel_error);
    mprintf("  Position uncertainty: %.2f m (1-sigma)\n", sqrt(state.covar(1,1)));
    mprintf("  Velocity uncertainty: %.2f m/s (1-sigma)\n", sqrt(state.covar(2,2)));
    mprintf("\n");
end

// Plot results
mprintf("Generating plots...\n");

scf(1);
clf();

// Define colors for each target
colors = ["b", "r", "g"];
markers = ["o", "s", "d"];

// Plot trajectories
subplot(1, 2, 1);
title("Target Trajectories");
xlabel("X Position (m)");
ylabel("Y Position (m)");

for i = 1:num_targets
    // Extract trajectory data
    x_truth = matrix(history_truth(i, 1, :), 1, num_steps);
    y_truth = matrix(history_truth(i, 2, :), 1, num_steps);
    x_est = matrix(history_est(i, 1, :), 1, num_steps);
    y_est = matrix(history_est(i, 2, :), 1, num_steps);
    x_meas = matrix(history_meas(i, 1, :), 1, num_steps);
    y_meas = matrix(history_meas(i, 2, :), 1, num_steps);

    // Plot truth (solid line)
    plot(x_truth, y_truth, colors(i) + "-");

    // Plot estimate (dashed line)
    plot(x_est, y_est, colors(i) + "--");

    // Plot measurements (dots)
    plot(x_meas, y_meas, colors(i) + ".");
end

legend(["Target 1 Truth", "Target 1 Est", "Target 1 Meas", ...
        "Target 2 Truth", "Target 2 Est", "Target 2 Meas", ...
        "Target 3 Truth", "Target 3 Est", "Target 3 Meas"]);

// Plot estimation errors over time
subplot(1, 2, 2);
title("Position Estimation Error");
xlabel("Time Step");
ylabel("Error (m)");

for i = 1:num_targets
    errors = zeros(1, num_steps);
    for t = 1:num_steps
        errors(t) = norm(history_truth(i, :, t) - history_est(i, :, t));
    end
    plot(1:num_steps, errors, colors(i) + "-o");
end

legend(["Target 1", "Target 2", "Target 3"]);

mprintf("\nDemo complete! Check the plot window.\n");
