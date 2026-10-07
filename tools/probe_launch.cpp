// tools/probe_launch.cpp - does a kernel launch work on each GPU, as an ordinary and as a cooperative kernel?
//
// Strata's SYCL port marked every kernel with use_root_sync, which makes DPC++ submit it as a cooperative kernel.
// On 2x Arc A770 (i915) the engine's first kernel on the second GPU failed with UR_RESULT_ERROR_OUT_OF_HOST_MEMORY.
// This runs the same shape (gather_rows: 384 work-groups of 256, a 16-byte row gather) on every Level Zero GPU,
// both ways, plus a USM host read from that GPU (the expert mirror's path), and prints what each one does.
//
//   ./arc probe        (builds this with icpx and runs it)
#include <sycl/sycl.hpp>
#include <cstdio>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

template <typename Props>
static std::string run(sycl::queue& q, Props props, const sycl::uint4* src, const int* ids, sycl::uint4* dst, long long n,
                       long long row) {
    try {
        q.parallel_for(sycl::nd_range<1>(384 * 256, 256), props, [=](sycl::nd_item<1> it) {
             for (long long i = it.get_global_id(0); i < n * row; i += (long long) it.get_global_range(0)) {
                 const long long r = i / row, o = i - r * row;
                 dst[i] = src[(long long) ids[r] * row + o];
             }
         }).wait_and_throw();
        return "ok";
    } catch (const sycl::exception& e) {
        return std::string("FAILED: ") + e.what();
    }
}

int main() {
    const long long rows = 4096, row = 2100 / 16 + 1;   // a Q6_K-sized head row, in 16-byte elements
    int failures = 0;
    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.empty()) {
        std::printf("no Level Zero GPU visible (ONEAPI_DEVICE_SELECTOR, the render group?)\n");
        return 2;
    }
    sycl::context ctx(gpus);   // one context for all of them, as dpct makes it
    for (size_t g = 0; g < gpus.size(); ++g) {
        sycl::queue q(ctx, gpus[g], sycl::property::queue::in_order{});
        std::printf("GPU %zu: %s, max_mem_alloc_size %.2f GiB\n", g, gpus[g].get_info<sycl::info::device::name>().c_str(),
                    (double) gpus[g].get_info<sycl::info::device::max_mem_alloc_size>() / 1073741824.0);
        auto* src = sycl::malloc_device<sycl::uint4>((size_t) (rows * row), q);
        auto* dst = sycl::malloc_device<sycl::uint4>((size_t) (rows * row), q);
        auto* ids = sycl::malloc_device<int>((size_t) rows, q);
        std::vector<int> h(rows);
        for (long long i = 0; i < rows; ++i) h[(size_t) i] = (int) (rows - 1 - i);
        q.memset(src, 0, (size_t) (rows * row) * sizeof(sycl::uint4)).wait();
        q.memcpy(ids, h.data(), h.size() * sizeof(int)).wait();
        const std::string plain = run(q, syclex::properties{}, src, ids, dst, rows, row);
        const std::string coop = run(q, syclex::properties{syclex::use_root_sync}, src, ids, dst, rows, row);
        // the mirror's path: a kernel on this GPU reading pinned host memory
        std::string host = "ok";
        auto* hsrc = sycl::malloc_host<sycl::uint4>((size_t) (rows * row), q);
        if (hsrc == nullptr) host = "FAILED: malloc_host";
        else {
            std::fill(hsrc, hsrc + rows * row, sycl::uint4{});
            host = run(q, syclex::properties{}, hsrc, ids, dst, rows, row);
            sycl::free(hsrc, q);
        }
        std::printf("  ordinary launch:              %s\n", plain.c_str());
        std::printf("  cooperative (use_root_sync):  %s\n", coop.c_str());
        std::printf("  reading pinned host memory:   %s\n", host.c_str());
        failures += (plain != "ok") + (host != "ok");
        sycl::free(src, q);
        sycl::free(dst, q);
        sycl::free(ids, q);
    }
    std::printf(failures ? "\nan ordinary launch failed: the problem is not the cooperative path\n"
                         : "\nordinary launches work on every GPU\n");
    return failures ? 1 : 0;
}
