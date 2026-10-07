// Real GGUF experts against ggml's float reference, including entry permutations and tile tails.
// Build with tools/build-prefill-parity.sh; run: prefill-parity <shard1.gguf> [layer...]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_prefill.hpp"
#include "ggml.h"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <vector>
namespace k = strata::kernels;
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: prefill-parity shard1.gguf [layer...]\n"); return 2; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    strata::GgufFile file(argv[1]);
    sycl::queue* q = &dpct::get_in_order_queue();
    constexpr int H = 2560, FF = 640, expert = 7;
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
    if (layers.empty()) {
        std::set<int> types;
        for (int l = 0; l < 48; ++l)
            for (const auto& ti : file.tensors())
                if (ti.name == "blk." + std::to_string(l) + ".ffn_gate_exps.weight" &&
                    types.insert((int)ti.type).second) layers.push_back(l);
    }
    int failures = 0;
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const char* roles[] = {"gate", "up", "down"};
        for (const auto& ti : file.tensors())
            for (int r = 0; r < 3; ++r)
                if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
        if (!t[0] || !t[1] || !t[2]) { ++failures; continue; }
        const auto L = k::native_expert_layout((int)t[0]->type, (int)t[2]->type, H, FF);
        std::vector<uint8_t> blob(L.bytes);
        std::memcpy(blob.data(), file.tensor_data(*t[0]) + expert * L.up_off, L.up_off);
        std::memcpy(blob.data() + L.up_off, file.tensor_data(*t[1]) + expert * L.up_off, L.up_off);
        std::memcpy(blob.data() + L.down_off, file.tensor_data(*t[2]) + expert * (L.bytes - L.down_off), L.bytes - L.down_off);
        std::vector<float> G(FF * H), U(FF * H), D(H * FF);
        const auto* gu = ggml_get_type_traits((ggml_type)L.gu_type);
        const auto* dn = ggml_get_type_traits((ggml_type)L.d_type);
        for (int r = 0; r < FF; ++r) {
            gu->to_float(blob.data() + r * L.gu_row, G.data() + r * H, H);
            gu->to_float(blob.data() + L.up_off + r * L.gu_row, U.data() + r * H, H);
        }
        for (int r = 0; r < H; ++r) dn->to_float(blob.data() + L.down_off + r * L.d_row, D.data() + r * FF, FF);
        auto* db = sycl::malloc_device<uint8_t>(blob.size(), *q);
        q->memcpy(db, blob.data(), blob.size()).wait();
        for (int nt : {7, 17, 65, 160}) {
            std::mt19937 rng(110 + l + nt);
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<float> x(nt * H), ref(nt * H), got(nt * H);
            for (auto& v : x) v = nd(rng);
            std::vector<int32_t> tok(nt), dst(nt);
            std::iota(tok.begin(), tok.end(), 0);
            std::shuffle(tok.begin(), tok.end(), rng);
            for (int e = 0; e < nt; ++e) dst[e] = nt - 1 - e;
            for (int e = 0; e < nt; ++e) {
                std::vector<float> hidden(FF);
                const float* a = x.data() + tok[e] * H;
                for (int r = 0; r < FF; ++r) {
                    double g = 0, u = 0;
                    for (int i = 0; i < H; ++i) { g += (double)G[r * H + i] * a[i]; u += (double)U[r * H + i] * a[i]; }
                    hidden[r] = (float)(g / (1.0 + std::exp(-g)) * u);
                }
                for (int r = 0; r < H; ++r) {
                    double value = 0;
                    for (int i = 0; i < FF; ++i) value += (double)D[r * FF + i] * hidden[i];
                    ref[dst[e] * H + r] = (float)value;
                }
            }
            auto* dx = sycl::malloc_device<float>(x.size(), *q);
            auto* xq = sycl::malloc_device<uint8_t>((size_t)nt * H / 32 * 36, *q);
            auto* scratch = sycl::malloc_device<uint8_t>(k::native_expert_scratch_bytes(nt, FF), *q);
            auto* out = sycl::malloc_device<float>(got.size(), *q);
            auto* ptr = sycl::malloc_shared<unsigned long long>(3, *q);
            auto* start = sycl::malloc_shared<int32_t>(4, *q);
            auto* ng = sycl::malloc_shared<int32_t>(1, *q);
            auto* dtok = sycl::malloc_shared<int32_t>(nt, *q);
            auto* ddst = sycl::malloc_shared<int32_t>(nt, *q);
            ptr[0] = ptr[1] = (unsigned long long)db;
            *ng = 2; start[0] = 0; start[1] = nt / 2 + 1; start[2] = nt;
            std::copy(tok.begin(), tok.end(), dtok);
            std::copy(dst.begin(), dst.end(), ddst);
            q->memcpy(dx, x.data(), x.size() * 4).wait();
            q->fill(out, NAN, got.size()).wait();
            // Flatten rows: the quantizer's optimized path accepts at most 8 columns.
            k::quantize_q8_1_rows(dx, 1, nt * H, xq, q);
            k::native_prefill_expert_grouped(L, ptr, start, ng, ddst, dtok, 3, nt, xq, scratch, out, q);
            q->wait_and_throw();
            q->memcpy(got.data(), out, got.size() * 4).wait();
            double err = 0, norm = 0;
            bool finite = true;
            for (size_t i = 0; i < got.size(); ++i) {
                finite &= std::isfinite(got[i]);
                err += std::fabs((double)got[i] - ref[i]);
                norm += std::fabs((double)ref[i]);
            }
            const double relative = err / (norm + 1e-30);
            const bool ok = finite && relative < 0.03; // same tolerance as upstream native_expert_parity
            std::printf("layer %d type %d/%d rows %d: float-reference L1 relative %.6g %s\n",
                        l, L.gu_type, L.d_type, nt, relative, ok ? "ok" : "FAIL");
            failures += !ok;
            sycl::free(dx, *q); sycl::free(xq, *q); sycl::free(scratch, *q); sycl::free(out, *q);
            sycl::free(ptr, *q); sycl::free(start, *q); sycl::free(ng, *q); sycl::free(dtok, *q); sycl::free(ddst, *q);
        }
        sycl::free(db, *q);
    }
    std::printf("prefill-parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
