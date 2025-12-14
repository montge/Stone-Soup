// Stone Soup Scilab Module Cleaner
// Run this script to clean/uninstall the Stone Soup Scilab bindings
//
// Usage:
//   exec("cleaner.sce", -1);

mode(-1);
lines(0);

// Get the directory containing this script
cleaner_path = get_absolute_file_path("cleaner.sce");

mprintf("Cleaning Stone Soup Scilab bindings...\n");
mprintf("Cleaner path: %s\n", cleaner_path);

// Clean gateway compiled files
mprintf("\n=== Cleaning gateway ===\n");
gateway_path = cleaner_path + "sci_gateway/";
gateway_files = [
    "loader_gateway.sce";
    "libstonesoup_gateway.so";
    "libstonesoup_gateway.dll";
    "libstonesoup_gateway.dylib";
    "cleaner_gateway.sce";
];
for i = 1:size(gateway_files, "*")
    f = gateway_path + gateway_files(i);
    if isfile(f) then
        mdelete(f);
        mprintf("  Deleted: %s\n", gateway_files(i));
    end
end

// Clean object files
obj_pattern = gateway_path + "*.o";
obj_files = listfiles(obj_pattern);
if ~isempty(obj_files) then
    for i = 1:size(obj_files, "*")
        mdelete(obj_files(i));
        mprintf("  Deleted: %s\n", obj_files(i));
    end
end

// Clean macros compiled files
mprintf("\n=== Cleaning macros ===\n");
macros_path = cleaner_path + "macros/";
macros_files = [
    "lib";
    "names";
];
for i = 1:size(macros_files, "*")
    f = macros_path + macros_files(i);
    if isfile(f) then
        mdelete(f);
        mprintf("  Deleted: %s\n", macros_files(i));
    end
end

// Clean .bin files (compiled macros)
bin_pattern = macros_path + "*.bin";
bin_files = listfiles(bin_pattern);
if ~isempty(bin_files) then
    for i = 1:size(bin_files, "*")
        mdelete(bin_files(i));
        mprintf("  Deleted: %s\n", bin_files(i));
    end
end

// Clean help compiled files
mprintf("\n=== Cleaning help ===\n");
help_path = cleaner_path + "help/";
if isdir(help_path) then
    help_compiled = help_path + "scilab_en_US_help/";
    if isdir(help_compiled) then
        rmdir(help_compiled, "s");
        mprintf("  Deleted: scilab_en_US_help/\n");
    end
end

// Clean loader script
mprintf("\n=== Cleaning loader ===\n");
loader_file = cleaner_path + "loader.sce";
if isfile(loader_file) then
    mdelete(loader_file);
    mprintf("  Deleted: loader.sce\n");
end

// Clean any generated documentation
jar_pattern = cleaner_path + "jar/";
if isdir(jar_pattern) then
    rmdir(jar_pattern, "s");
    mprintf("  Deleted: jar/\n");
end

mprintf("\n=== Clean complete ===\n");
mprintf("To rebuild, run: exec(""%sbuilder.sce"", -1);\n", cleaner_path);
