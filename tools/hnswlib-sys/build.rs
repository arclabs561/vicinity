fn main() {
    // Build the vendored hnswlib headers (header-only) plus the `extern "C"`
    // shim. hnswlib needs C++11 at minimum; C++14 avoids the constructs its
    // newer Status/StatusOr code guards on `__cplusplus`.
    cc::Build::new()
        .cpp(true)
        .file("src/hnswlib_shim.cpp")
        .include("vendor")
        .flag_if_supported("-std=c++14")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("hnswlib_shim");

    println!("cargo:rerun-if-changed=src/hnswlib_shim.cpp");
    for header in [
        "bruteforce.h",
        "hnswalg.h",
        "hnswlib.h",
        "space_ip.h",
        "space_l2.h",
        "stop_condition.h",
        "visited_list_pool.h",
    ] {
        println!("cargo:rerun-if-changed=vendor/hnswlib/{header}");
    }
}
