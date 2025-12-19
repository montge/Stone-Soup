// Stone Soup Xcos Palette Loader
//
// This script loads the Stone Soup Xcos palette into Scilab.
// Run this script after loading the Stone Soup toolbox.
//
// Usage:
//   exec("loader.sce");
//
// The palette will appear in the Xcos palette browser under "Stone Soup".

// Get the directory containing this script
xcos_dir = get_absolute_file_path("loader.sce");

// Load the Xcos block functions
exec(xcos_dir + "STONESOUP_KALMAN_PREDICT.sci");
exec(xcos_dir + "STONESOUP_KALMAN_UPDATE.sci");
exec(xcos_dir + "STONESOUP_CONSTANT_VELOCITY.sci");

// Create the palette
pal = xcosPal("Stone Soup");

// Add blocks to palette
pal = xcosPalAddBlock(pal, "STONESOUP_KALMAN_PREDICT", xcos_dir + "icons/kalman_predict.svg");
pal = xcosPalAddBlock(pal, "STONESOUP_KALMAN_UPDATE", xcos_dir + "icons/kalman_update.svg");
pal = xcosPalAddBlock(pal, "STONESOUP_CONSTANT_VELOCITY", xcos_dir + "icons/constant_velocity.svg");

// Add palette to Xcos
xcosPalAdd(pal);

disp("Stone Soup Xcos palette loaded successfully.");
disp("Open Xcos (menu Simulation -> Open in Xcos) to see the palette.");
