fn main() {
    // Build the vendored Annoy C++ header library (header-only: `annoylib.h`)
    // plus the `extern "C"` shim. No system packages are required; Apple clang
    // supplies libc++.
    cc::Build::new()
        .cpp(true)
        .file("src/annoy_shim.cpp")
        .include("vendor")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("annoy_shim");

    println!("cargo:rerun-if-changed=src/annoy_shim.cpp");
    println!("cargo:rerun-if-changed=vendor/annoylib.h");
    println!("cargo:rerun-if-changed=vendor/kissrandom.h");
}
