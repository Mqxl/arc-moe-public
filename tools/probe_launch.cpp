// tools/probe_launch.cpp - does a kernel launch work on each GPU, as an ordinary and as a cooperative kernel?
//
// Strata's SYCL port marked every kernel with use_root_sync, which makes DPC++ submit it as a cooperative kernel.
// On 2x Arc A770 (i915) the engine's first kernel on the second GPU failed with UR_RESULT_ERROR_OUT_OF_HOST_MEMORY.
// This runs the same shape (gather_rows: 384 work-groups of 256, a 16-byte row gather) on every Level Zero GPU,
// both ways, plus a USM host read from that GPU (the expert mirror's path), and prints what each one does.
// It also checks the free-VRAM query: with 2 GiB allocated, the driver's answer against what /proc/self/fdinfo says
// this process holds (on i915 an unprivileged process is told the whole card is free).
//
//   ./arc probe        (builds this with icpx and runs it)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <sycl/sycl.hpp>
#include <cctype>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <sstream>
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

// the VRAM this process holds on the device at `pci`, from the DRM fdinfo memory stats; -1 when the kernel prints none
static long long fdinfo_vram(std::string pci) {
    for (auto& c : pci) c = (char) std::tolower((unsigned char) c);
    DIR* dir = opendir("/proc/self/fdinfo");
    if (!dir) return -1;
    long long sum = -1;
    while (dirent* e = readdir(dir)) {
        if (e->d_name[0] == '.') continue;
        std::ifstream f(std::string("/proc/self/fdinfo/") + e->d_name);
        std::string line, pdev;
        long long mem = -1;
        while (std::getline(f, line)) {
            if (line.rfind("drm-pdev:", 0) == 0) {
                std::istringstream is(line.substr(9));
                is >> pdev;
                for (auto& c : pdev) c = (char) std::tolower((unsigned char) c);
            } else if (line.rfind("drm-total-local", 0) == 0 || line.rfind("drm-total-vram", 0) == 0) {
                std::istringstream is(line.substr(line.find(':') + 1));
                unsigned long long v = 0;
                std::string unit;
                if (!(is >> v)) continue;
                is >> unit;
                const unsigned long long mul = unit == "KiB" ? 1ull << 10 : unit == "MiB" ? 1ull << 20 : unit == "GiB" ? 1ull << 30 : 1;
                mem = (mem < 0 ? 0 : mem) + (long long) (v * mul);
            }
        }
        if (mem >= 0 && pdev == pci) sum = (sum < 0 ? 0 : sum) + mem;
    }
    closedir(dir);
    return sum;
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
        // the free-VRAM query, with 2 GiB held by this process
        {
            const size_t big = 2ull << 30;
            void* hold = sycl::malloc_device(big, q);
            if (hold) q.memset(hold, 0, big).wait();
            const double total = (double) gpus[g].get_info<sycl::info::device::global_mem_size>() / 1073741824.0;
            double drv = -1;
            if (gpus[g].has(sycl::aspect::ext_intel_free_memory))
                drv = (double) gpus[g].get_info<sycl::ext::intel::info::device::free_memory>() / 1073741824.0;
            long long used = -1;
            if (gpus[g].has(sycl::aspect::ext_intel_pci_address))
                used = fdinfo_vram(gpus[g].get_info<sycl::ext::intel::info::device::pci_address>());
            std::printf("  free VRAM with 2 GiB held:    driver says %.2f of %.2f GiB; fdinfo: this process holds %s\n",
                        drv, total, used < 0 ? "(no stats)" : (std::to_string(used >> 20) + " MiB").c_str());
            if (drv >= total - 0.05) std::printf("  -> the driver reports the whole card free: it does not see allocations\n");
            if (hold) sycl::free(hold, q);
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
