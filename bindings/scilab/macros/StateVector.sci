// StateVector.sci - State vector operations for Stone Soup
//
// This file provides high-level Scilab functions for working with
// state vectors in the Stone Soup tracking framework.

function sv = StateVector(varargin)
    // Create a state vector
    //
    // Calling Sequence
    //   sv = StateVector(dim)
    //   sv = StateVector(dim, value)
    //   sv = StateVector(data)
    //
    // Parameters
    //   dim : scalar, dimension of the state vector
    //   value : scalar, fill value (default 0)
    //   data : vector, initial data
    //   sv : column vector, the state vector
    //
    // Description
    //   Creates a state vector for use in Stone Soup tracking operations.
    //   State vectors are represented as column vectors in Scilab.
    //
    // Examples
    //   // Create a 4-dimensional zero state vector
    //   sv = StateVector(4);
    //
    //   // Create a 4-dimensional state vector filled with 1.0
    //   sv = StateVector(4, 1.0);
    //
    //   // Create a state vector from existing data
    //   sv = StateVector([1; 0.5; 2; 0.3]);

    rhs = argn(2);

    if rhs == 0 then
        error("StateVector requires at least one argument");
    end

    arg1 = varargin(1);

    if size(arg1, '*') == 1 then
        // Scalar: create vector of given dimension
        dim = int(arg1);
        if dim <= 0 then
            error("Dimension must be positive");
        end

        if rhs >= 2 then
            value = varargin(2);
        else
            value = 0.0;
        end

        sv = stonesoup_state_vector_create(dim, value);
    else
        // Vector: use as initial data
        sv = arg1(:);  // Ensure column vector
    end
endfunction

function n = sv_norm(sv)
    // Compute Euclidean norm of state vector
    //
    // Calling Sequence
    //   n = sv_norm(sv)
    //
    // Parameters
    //   sv : state vector
    //   n : Euclidean norm
    //
    // Examples
    //   sv = StateVector([3; 4]);
    //   n = sv_norm(sv);  // Returns 5

    n = stonesoup_state_vector_norm(sv);
endfunction

function result = sv_add(sv1, sv2)
    // Add two state vectors
    //
    // Calling Sequence
    //   result = sv_add(sv1, sv2)
    //
    // Parameters
    //   sv1 : first state vector
    //   sv2 : second state vector
    //   result : sum of the two vectors
    //
    // Examples
    //   sv1 = StateVector([1; 2]);
    //   sv2 = StateVector([3; 4]);
    //   result = sv_add(sv1, sv2);  // Returns [4; 6]

    result = stonesoup_state_vector_add(sv1, sv2);
endfunction

function result = sv_subtract(sv1, sv2)
    // Subtract two state vectors
    //
    // Calling Sequence
    //   result = sv_subtract(sv1, sv2)
    //
    // Parameters
    //   sv1 : first state vector
    //   sv2 : second state vector
    //   result : difference of the two vectors

    result = sv1 - sv2;
endfunction

function result = sv_scale(sv, factor)
    // Scale a state vector
    //
    // Calling Sequence
    //   result = sv_scale(sv, factor)
    //
    // Parameters
    //   sv : state vector
    //   factor : scaling factor
    //   result : scaled vector

    result = factor * sv;
endfunction
