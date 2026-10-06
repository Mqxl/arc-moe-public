# Engineering notes: Strata's SYCL port on Arc A770 (Alchemist)

How each problem was found, what the patch changes and why it is shaped that way. The patches are in
`patches/strata/`, against Strata `a5682fe`; `./arc build` applies them to `third_party/strata` on the branch
`arc-moe`.

## 0. Baseline: which Strata

The SYCL port (`sycl/`) is a dpct migration of the CUDA engine: `src/X.cu` became `sycl/src/X.dp.cpp`, and host
files that called the CUDA runtime were copied. Upstream keeps changing the shared headers in `include/` without
re-migrating the copies, so the port breaks silently on NVIDIA-only CI.

Compiling every SYCL translation unit at each candidate commit (device-only, `tools/check_fp64.py` reports a failed
compile as `??`):

| Commit | Result |
|---|---|
| `82f46a8` = `v0.1.40.1` = `main` (2026-10-06) | 3 TUs fail: `fused_gr.dp.cpp` (`fused_gr_read_multi` returns `bool` in the header), `expert_source.cpp` (`load_experts_gguf` ambiguous), `gemm.dp.cpp` (`Gemm::bf16` / `Gemm::native` signatures) |
| `1cbcacb` = `v0.1.40` | same 3 failures |
| `c8c7e21` (last commit touching `sycl/`) | same 3 failures |
| **`a5682fe`** | **compiles** |

`git bisect --first-parent` between `a5682fe` (good) and `c8c7e21` (bad) names `bc6b6f8`, "Merge the Aurora
gfx1151 branch ... onto 0.1.39", the commit right after `a5682fe`. `a5682fe` is 0.1.39 plus the two SYCL fixes
(the #559/#626 compile fix and the `cudaStreamQuery` ring-wait fix): the exact tree of the 2x Arc Pro B60
community report (IQ2_XS 55-61 tok/s). It is the pin.

## 1. FP64 on a GPU without FP64

Xe-HPG (DG2: A770, A750, A580, A380) has no double-precision unit. Upstream accumulates several reductions in
`double` on purpose (byte-exact parity with `ref/*.py`, which works in float64), and Battlemage runs that in
hardware, so the port kept it.

The proof that it breaks A770 is one small kernel:

```
$ icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device acm-g10" kd.cpp
error: Double type is not supported on this platform.
$ ... "-device bmg-g21" kd.cpp
Build succeeded.
```

A JIT build (SPIR-V) compiles, but the runtime refuses the kernel at its first launch: the device lacks
`aspect::fp64`. With `OverrideDefaultFP64Settings=1` / `IGC_EnableDPEmulation=1` the driver emulates FP64 in
software, and a router softmax over 512 experts in emulated double, 48 layers per token, is exactly the latency
the decode path cannot afford.

**Finding every use.** grep is not enough: `1.0 + sycl::exp(-x)` with `float x` is double arithmetic, and the
dpct sub-group helpers get instantiated for double through templates. `tools/check_fp64.py` compiles each TU
with `-fsycl-device-only -fsycl-device-obj=spirv`, disassembles with oneAPI's `llvm-spirv -to-text`, gathers
the 64-bit `TypeFloat` ids (and the vectors and arrays of them), and counts, per SPIR-V function, the
instructions whose result type is one of them. Before the patch:

```
FP64 router_top10.dp.cpp      router_top10 kernel                                  18
FP64 quantize_act.dp.cpp      quantize_q8_0                                         6
FP64 sampler.dp.cpp           coupled_draft_sample 31, sampled_tail_warp 30, sample_tokens 26
FP64 shared_expert.dp.cpp     shared_expert(_multi) 18+18+5, moe_combine 7
FP64 gdn.dp.cpp               gdn_out_norm 13, gdn_l2_norm 11
FP64 qsa.dp.cpp               indexer_key_append 32, qsa_index_step 14, qsa_gate_apply(_f32) 7+7
FP64 elementwise.dp.cpp       silu_inplace 5
FP64 ple.dp.cpp               block_sum 11, ple_block x5
```

These are hot paths: the router and the GDN/QSA norms run in every layer of every token.

**The change.** `sycl/include/strata/sycl_fp.hpp` defines `sfp_t`, `float` under `STRATA_SYCL_NO_FP64` and
`double` otherwise. In those device functions `double` becomes `sfp_t` and floating literals become
`sfp_t(...)`, so no implicit promotion is left. That includes the work-group local arrays
(`group_local_memory_for_overwrite<sfp_t[8]>`), the dynamic local memory sizing (`sizeof(sfp_t)`) and the dpct
shuffles. CMake turns the option on whenever `STRATA_SYCL_AOT` names an Alchemist part (`acm-*`, `dg2`), so a
Battlemage build stays bit-for-bit upstream. After the patch: **84 TUs, 0 with FP64**, and the full AOT build for
`acm-g10` compiles all 1001 device modules.

**Precision.** Each reduction is a tree over 32-lane sub-groups, or at most 512 positive terms bounded by 1 (the
router's softmax sum, where the largest term is exactly `exp(0)`). The result is rounded to `float` right after.
llama.cpp's SYCL and CUDA backends do the same reductions in float. What changes is the last bits of a probability
or a norm, not the routing (the top-10 selection compares the float `p[e]` as before).

## 2. Buffers larger than 4 GB

The expert cache is a single `malloc_device` of 10-15 GB. On DG2 `max_mem_alloc_size` is about 4 GB:

- Level Zero refuses the allocation unless `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` (the runtime then passes
  `ze_relaxed_allocation_limits_exp_desc_t`). The native runner and the config set it.
- IGC may assume a buffer under 4 GB and address it with 32-bit offsets.
  `-ze-opt-greater-than-4GB-buffer-required` keeps stateless 64-bit addressing in every kernel. It goes to `ocloc`
  in an AOT build (`-options "..."`) and to the JIT through `-Xsycl-target-backend=spir64`, next to the existing
  `-cl-fp32-correctly-rounded-divide-sqrt`.

## 3. The expert mirror with a layer split

IQ2_XS has 35.5 GB of experts and two A770s hold about 26 GB of them. The port's answer to "more experts than
VRAM" is the pinned host mirror (`GgufExpertSource::mirror`): the experts that did not fit go once into
page-locked RAM, and the device-built verify plan reads them over PCIe inside the decode window. That was measured
at 50-60 tok/s on a single 32 GB B70.

In `generate.cpp`, with `--layer-split`, `profile` is already CUDA0's share when the mirror is built. But the
second loop ("pairs the profile does not list at all") walked **all 48 layers**, so every expert of the later
stage, which that stage's own cache holds, counted as missing:

- the RAM budget went to experts nobody reads from the mirror (only CUDA0's plan has a mirror table);
- CUDA0's real misses, the least-routed ones that come last in profile order, were left out once the budget ran out,
  and went to the SSD;
- the "N experts are neither in VRAM nor mirrored" count included the other card's layers. That is why the B60
  notes say to set `STRATA_MIRROR_MIB=0` with a split.

The patch limits both loops to `[0, split_at[0])` and replaces the `std::find` over the miss list (quadratic,
about 10^8 comparisons) with a bitmap.

**Placement.** `--layer-split auto` searches the boundary with a cost model: per-layer compute plus a miss cost of
190 ms × the unheld routed mass. With the mirror in place, a CUDA0 miss is a PCIe read, while a later stage's
miss still takes the host path. With `STRATA_SPLIT_MIRROR_MS` set (25 in the config), the model prices the two
separately. The search then puts the boundary where the second card holds every expert of its layers and CUDA0
carries the overflow into the mirror.

## 4. `STRATA_VERIFY_NO_HOST=auto`

`NO_HOST=1` makes the host wait for the whole verify window instead of each layer's doorbell, because on Arc the
GPU's writes to host-mapped memory are not reliably visible while a graph runs. Upstream's Docker runner always
sets it. It is only valid when no expert needs the host. If one does, the GPU waits for a plan that never comes,
which is the hang seen on small cards.

`auto` is resolved in `generate.cpp` after the placement: `1` when CUDA0 has no unmirrored miss and every later
stage holds all `(layer, expert)` pairs of its layers, otherwise unset with a message that says why.
`verify.cpp` now reads the variable per window instead of caching it in a function-local static, so the decision
holds even if a window ran earlier during warm-up.

## 5. Native runner, images, two cards (`sycl/setup_intel.py`)

- `STRATA_SYCL_NATIVE=1` (the default when the `strata-sycl-dev` image does not exist): the config's `exe` is
  `sycl/serve/strata-sycl-native.sh`, which sources `setvars.sh` and execs the engine. Paths stay host paths.
- `arc_env()` writes the engine's switches into the config: device plan, `NO_HOST=auto`, 12 stager threads,
  `ZES_ENABLE_SYSMAN`, relaxed allocations, `STRATA_SPLIT_MIRROR_MS` and `ONEAPI_DEVICE_SELECTOR=arc`, which the runner resolves through `sycl-ls` to the discrete Arcs only: an Intel CPU's integrated GPU would otherwise be `level_zero:0` and take a card's place in the split.
- Images: the engine side (`--vision`, `GENI` requests with `strata-vision` embedding records, the mrope table on
  every stage) is already in the SYCL port. `setup_intel.py` used to force `--vision none` and drop the config's
  `vision` section. Now it runs setup.py's own #304 path, which compiles `tools/vision` (llama.cpp mtmd, CPU only)
  into `engine-sycl/` and writes the `vision` section with `threads = cores/2`.
- With two or more Arcs, setup gets `--gpus 0,1`: the config's `"gpu": [0, 1]` and `"layer_split": "auto"`, and
  the server adds `--layer-split auto`.
- VRAM reserve 1536 MiB up to 32K of context (upstream 1024). The i915 and xe drivers start moving buffers to
  RAM when VRAM runs out, and the B60 report ran with 1536.

## 6. IGC crashes on the XMX kernels for acm-g10

With FP64 gone, the AOT link of the engine compiles 1001 device modules (one per kernel). On `acm-g10`, 994 build
and 7 crash the compiler itself:

```
IGC: Internal Compiler Error: Floating point exception
Build failed with error code: -11
```

To find the 7, the link ran again with `-save-temps` and was stopped once the per-kernel SPIR-V files existed. Each
one then went through `ocloc compile -device acm-g10` (4 in parallel), and each was disassembled to see what it
uses. The 7 failures are exactly the 7 modules with the cooperative-matrix capability: six instances of
`qsa_prompt_attn_xmx`'s `launch<KV, CH>` and `xmx_gemm_iq`. All of them use 16x16x16 FP16 `joint_matrix` tiles,
which is Xe2's DPAS shape. Alchemist's DPAS takes M <= 8, N = 8, K = 16 for FP16.

Both kernels are opt-in upstream (`STRATA_PROMPT_ATTN_XMX=1`, the `xmx_gemm_bench`), and `docs/INTEL.md` measured
both slower than their fallbacks on the B70. `STRATA_SYCL_NO_XMX` (on by default for an Alchemist target) builds
them as stubs that return `false`. The callers already treat `false` as "not here" and take the FP32 prompt
attention and the dequantize + oneMKL GEMM.

## 7. `a5682fe` does not link either

The device-only check compiles every TU, but the host link of `a5682fe` fails:

```
undefined reference to `strata::kernels::resident_plan(int const*, ..., void*, unsigned int*)'
```

`6f7f680` (#871) gave `resident_plan` a 14th parameter, `uint32_t* plan_err = nullptr`, in the shared
`include/strata/kernels/verify_kernels.hpp` and in the CUDA kernel. The SYCL definition kept 13. The SYCL
`verify.cpp` calls it with 13 arguments, which resolve through the default to the 14-parameter declaration that
nothing defines. (The 2x B60 report ran `6f32ec0`, a commit that is no longer on `main` after a rebase.) The patch
gives the SYCL definition the parameter and #871's behaviour when there is no skip flag: an empty plan and the error
flag with a system-scope fence, instead of a null store.

## What was verified here, and what was not

| Check | Status |
|---|---|
| Host + device compile of all 84 SYCL TUs at `a5682fe` + patches | done (icpx 2026.1.1) |
| FP64 scan of the device code, `STRATA_SYCL_NO_FP64=ON` | 0 instructions |
| Every per-kernel SPIR-V module through `ocloc -device acm-g10` | 994 of 1001 before patch 6; the 7 failures are exactly the `joint_matrix` modules |
| Host link of the engine | fails at `a5682fe` (#871 drift); links with patch 5 |
| AOT build of `strata` for `acm-g10` with all 6 patches | see the end of this table |
| A tiny double kernel: `acm-g10` fails, `bmg-g21` builds | reproduced |
| Run on an A770 | **not yet**: no Arc in the build machine. `./arc bench` reports the real numbers |
| Kernel parity tests (`ARC_PARITY=ON`, `ctest`) on an A770 | not yet. Expect the tight-tolerance ones that compare against float64 references to report ulp differences |
