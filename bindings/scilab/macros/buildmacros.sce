// Stone Soup Macros Builder
// This script builds the Stone Soup Scilab macros library

mode(-1);

macros_path = get_absolute_file_path("buildmacros.sce");

mprintf("Building Stone Soup macros...\n");

// List of macro files
macro_files = [
    "StateVector.sci";
    "GaussianState.sci";
    "kalman.sci";
];

// Generate binary library
try
    genlib("stonesouplib", macros_path, %f, macro_files);
    mprintf("Macros library built successfully.\n");
catch
    mprintf("Note: genlib not available, using exec-based loading.\n");
    // Create simple loader
    mputl(["// Stone Soup Macros Loader";
           "mode(-1);";
           "macros_path = get_absolute_file_path(""loadmacros.sce"");";
           "exec(macros_path + ""StateVector.sci"", -1);";
           "exec(macros_path + ""GaussianState.sci"", -1);";
           "exec(macros_path + ""kalman.sci"", -1);"],
           macros_path + "loadmacros.sce");
end
