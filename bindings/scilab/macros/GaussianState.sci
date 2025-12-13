// GaussianState.sci - Gaussian state operations for Stone Soup
//
// This file provides high-level Scilab functions for working with
// Gaussian states (state + covariance) in the Stone Soup tracking framework.

function gs = GaussianState(state_vector, covariance, timestamp)
    // Create a Gaussian state
    //
    // Calling Sequence
    //   gs = GaussianState(state_vector, covariance)
    //   gs = GaussianState(state_vector, covariance, timestamp)
    //
    // Parameters
    //   state_vector : column vector, the mean state
    //   covariance : square matrix, the covariance matrix
    //   timestamp : scalar, optional timestamp
    //   gs : tlist, Gaussian state structure
    //
    // Description
    //   Creates a Gaussian state representation with mean and covariance.
    //   The Gaussian state is represented as a typed list (tlist) with
    //   fields 'sv' (state vector), 'covar' (covariance), and 'ts' (timestamp).
    //
    // Examples
    //   // Create a 2D Gaussian state
    //   sv = [0; 0];
    //   P = eye(2, 2);
    //   gs = GaussianState(sv, P);
    //
    //   // Create with timestamp
    //   gs = GaussianState(sv, P, 0.0);

    rhs = argn(2);

    if rhs < 2 then
        error("GaussianState requires state_vector and covariance");
    end

    sv = state_vector(:);  // Ensure column vector
    dim = length(sv);

    // Validate covariance dimensions
    [r, c] = size(covariance);
    if r ~= dim | c ~= dim then
        error("Covariance matrix dimensions must match state vector dimension");
    end

    // Set timestamp
    if rhs >= 3 then
        ts = timestamp;
    else
        ts = 0.0;
    end

    // Create typed list
    gs = tlist(["GaussianState", "sv", "covar", "ts"], sv, covariance, ts);
endfunction

function sv = gs_state_vector(gs)
    // Get state vector from Gaussian state
    //
    // Calling Sequence
    //   sv = gs_state_vector(gs)
    //
    // Parameters
    //   gs : Gaussian state
    //   sv : state vector

    if typeof(gs) ~= "GaussianState" then
        error("Argument must be a GaussianState");
    end
    sv = gs.sv;
endfunction

function P = gs_covariance(gs)
    // Get covariance from Gaussian state
    //
    // Calling Sequence
    //   P = gs_covariance(gs)
    //
    // Parameters
    //   gs : Gaussian state
    //   P : covariance matrix

    if typeof(gs) ~= "GaussianState" then
        error("Argument must be a GaussianState");
    end
    P = gs.covar;
endfunction

function ts = gs_timestamp(gs)
    // Get timestamp from Gaussian state
    //
    // Calling Sequence
    //   ts = gs_timestamp(gs)
    //
    // Parameters
    //   gs : Gaussian state
    //   ts : timestamp

    if typeof(gs) ~= "GaussianState" then
        error("Argument must be a GaussianState");
    end
    ts = gs.ts;
endfunction

function dim = gs_dim(gs)
    // Get dimension of Gaussian state
    //
    // Calling Sequence
    //   dim = gs_dim(gs)
    //
    // Parameters
    //   gs : Gaussian state
    //   dim : dimension

    if typeof(gs) ~= "GaussianState" then
        error("Argument must be a GaussianState");
    end
    dim = length(gs.sv);
endfunction
