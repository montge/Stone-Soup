// Stone Soup Macros Loader
// This script loads all Stone Soup Scilab macros

mode(-1);

macros_path = get_absolute_file_path("loadmacros.sce");

// Load macro files
exec(macros_path + "StateVector.sci", -1);
exec(macros_path + "GaussianState.sci", -1);
exec(macros_path + "kalman.sci", -1);
