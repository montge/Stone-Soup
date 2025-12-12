use std::env;
use std::path::PathBuf;

fn main() {
    // Get the directory where libstonesoup will be located
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");

    // Tell cargo to look for shared libraries in the specified directory
    // This will be updated once the C library is built
    if let Ok(lib_path) = env::var("STONESOUP_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", lib_path);
    }

    // Link to libstonesoup once it's available
    // For now, this is a placeholder
    // Uncomment when C library is ready:
    // println!("cargo:rustc-link-lib=stonesoup");

    // Add rpath for dynamic linking on Unix systems
    #[cfg(target_family = "unix")]
    {
        if let Ok(lib_path) = env::var("STONESOUP_LIB_PATH") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);
        }
    }

    // Windows-specific configuration
    #[cfg(target_family = "windows")]
    {
        // Add Windows-specific link arguments if needed
    }

    println!("cargo:warning=Stone Soup C library not yet available - build will succeed but runtime linking may fail");
}
