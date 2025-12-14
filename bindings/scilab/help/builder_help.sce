// Stone Soup Scilab Help Builder
// This script builds the help documentation for the Stone Soup module

mode(-1);
lines(0);

help_path = get_absolute_file_path("builder_help.sce");
mprintf("Building Stone Soup help files...\n");
mprintf("Help path: %s\n", help_path);

// Build help for English
help_en_path = help_path + "en_US/";
if isdir(help_en_path) then
    mprintf("Building English help from: %s\n", help_en_path);

    try
        // Get the module path (parent of help directory)
        module_path = fullfile(help_path, "..");

        // Build help
        xmltojar(help_en_path, "Stone Soup", "en_US");

        mprintf("Help build successful.\n");
    catch
        mprintf("Warning: Help build failed. This is optional.\n");
        mprintf("Error: %s\n", lasterror());
    end
else
    mprintf("Warning: en_US help directory not found.\n");
end

mprintf("Help build complete.\n");
