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

## 8. First runs on 2x A770: what the hardware showed

These came from the first runs on the target machine (2x A770 16 GB, i915, 48 GB RAM, IQ2_XS at 32K).

- **The setup crashed** (`write_run_script() takes 3 positional arguments but 4 were given`). setup.py at the pin
  passes `open_browser`, and upstream fixed `setup_intel.py` for it only after the pin (#870). The wrapper now
  forwards every argument.
- **The engine stopped at `unknown argument: --remote-expert-opt`.** setup.py adds #578's switch to every
  multi-GPU config, at setup and again on its start path, and server.py adds `--batch N` for `"parallel"`. The SYCL
  engine parses neither. The config drops them and writes `"remote_expert_opt": false`, and the native runner strips
  them from the command line as well.
- **The mirror was refused** (`no pinned host memory for 10496 MiB`). It was one `malloc_host`, and A770's
  `max_mem_alloc_size` is ~4 GB; the relaxed limits cover device memory only. It is now 1 GiB chunks. On the next
  run: `7721 of 7721 experts ... mirrored (10.25 GiB, 4.8 s)` and `STRATA_VERIFY_NO_HOST=auto -> on`.
- **`UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` at `mtp.cpp:551`**, with 45 GiB of RAM free. The line is the catch of
  `MtpDrafter::bind`, not of the drafter's load. Labelled steps (patch 8) placed it at the `gather_rows` kernel on
  GPU 1. The wait on that device's earlier work right before it was clean. `gather_rows` is the first kernel the
  engine launches on the second GPU; until then that GPU only received allocations and copies.

  dpct gave all 318 kernel launches `use_root_sync`, so DPC++ submits every kernel as a **cooperative** kernel.
  No kernel uses a root-group barrier. The 2x B60 that ran this code were on the `xe` driver; the A770s are on
  `i915`. Patch 9 makes the property a macro that is empty on Alchemist, so the same kernels launch the ordinary way.
  `./arc probe` checks this hypothesis in a minute: it launches the same 384 x 256 gather on every GPU, both ways, plus
  a kernel that reads pinned host memory.

- **`./arc probe` ruled the cooperative path out.** Ordinary and cooperative launches both work on both cards.
  The real cue was in the engine's own log: `layer split, CUDA1: 15.11 GiB free of 15.11`, printed when GPU 1
  already held its layers' weights, the LM head, the session and the 839 MiB drafter. GPU 0 also read 15.11 free
  after loading 1.47 GiB. The free-VRAM query (`ext_intel_free_memory`, Level Zero Sysman) asks i915 for the
  region's `unallocated_size`, and i915 hides that from a process without `CAP_PERFMON`, so it answers with the
  probed size. Every free-memory query said "all free", the caches were sized for empty cards, and GPU 1 ended up
  overcommitted. i915 lets that happen, because local memory is backed lazily, and then fails the first execution
  that needs it all resident: the first kernel on GPU 1, `gather_rows`, with `UR_RESULT_ERROR_OUT_OF_HOST_MEMORY`.
  The probe's kernels did not show it because they allocate little. The B60s ran on `xe`, where the query works.

  Patch 10: `dpct::device_ext::get_memory_info` also reads `/proc/self/fdinfo` (`drm-pdev`, `drm-total-local*`
  on i915, `drm-total-vram*` on xe). The kernel prints these per DRM client, and a process can always read its
  own. Free memory becomes the smaller of the driver's answer and "total minus what this process holds".
  `./arc probe` now prints both numbers with 2 GiB allocated on each card.

- **Then the first prompt hung the GPU** (`verify: timed out at layer 0 ... (#267)`, and in the kernel log
  `i915 ... GPU HANG ... context reset`). With the VRAM count fixed, the split moved to K=37 and GPU 1's cache held
  5579 of its 5632 pairs: 53 short, because the split search's estimate is a few slots off once the drafter and the
  head land after it. One unheld pair turned `NO_HOST=auto` off. The window then took the per-layer host path, where
  the GPU spin-waits on host flag stores it does not reliably see (upstream's #667), and i915's hang check resets a
  kernel that spins that long.

  Patch 11: a later stage mirrors the pairs its cache could not hold. The pinned host memory is allocated in that
  device's own SYCL context, since dpct gives every device one and CUDA0's mirror is not usable from GPU 1. The stage
  also gets its own address table, and `resident_plan`'s mirror is now a registry per residency table. `NO_HOST=auto`
  then counts the stage as covered and stays on.

- **With the window host-free, GPU 0 still hung on the first prompt**, and it hung the same way without the window
  graph (`STRATA_VERIFY_EAGER=1`) and with the serial plan (`STRATA_PLAN_PARALLEL=0`). The kernel log shows only
  `GPU HANG ... context reset`, with no CAT error. That rules out the graph, and it rules out the flag waits as well:
  every device spin is bounded at `kSpinMax` = 20 000 reads, a few tens of ms. What is left is a kernel that does
  not finish, or one that reads an address the GPU has not mapped. On DG2 under i915 the second also shows up as a
  plain hang.

  Patch 12 adds `STRATA_VERIFY_STEPLOG=1` (with `STRATA_VERIFY_EAGER=1`). Every stage stamp of the window waits for
  the GPU and logs `verify step: GPU g layer l stage s done`. A hang ends with
  `FAILED in layer l before stage s ... last done: ...`, which names the kernels between the two stamps.
  After each `resident_plan` the device plan is printed: `skip` (planned, or refused, meaning a host fallback that
  never comes in a host-free window), its counts, and every group's expert pointer, marked `vram:` or `HOST:`.

- **The step log named it: layer 0, after stage 16, before `resident_plan` finished.** Between those two points
  only the router runs. `route<NE>` (`native_router.dp.cpp`, from llama.cpp's `topk-moe.cu`) launches an 8x32
  block, returns rows 1-7 at once (`if (local_id(1) != 0) return;`), and then calls a work-group barrier, which is
  dpct's translation of a `__syncthreads()`. On NVIDIA, exited warps don't hold a barrier. On Alchemist the
  barrier waits for every thread of the work-group, the exited ones included, so the kernel never ends and i915
  resets the GPU. The single-token decode path calls the same router, so every token would have hung too.

  Patch 13 removes the barrier. The router's values are registers and it uses no local memory, so the barrier had
  nothing to order. A scan of every SYCL kernel for an early `return` before a barrier found 30 more. All of them
  are uniform over the work-group: they test the group index or a value the whole group reads, so the whole group
  leaves together.

## 9. First measurements on the A770s, and where the time goes

`bb19834`, IQ2_XS, 32K context: decode 16.8 tok/s on code and 11.9 on prose, prompt reading ~50 tok/s. The
prompt rate is flat from 2K to 10K tokens.

**Decode.** Eager verify windows with a host clock at every stage (`./arc profile`, run B) average 400 ms on GPU 0
(37 layers) and 113 ms on GPU 1 (11 layers and the head). Real graph windows take about 55% of that. Per layer, a
trivial stage costs ~0.3 ms (the host's wait). Four stages cost 5-7x more on both cards:

| stage (per GDN layer, eager) | GPU 0 | GPU 1 | what runs |
|---|---|---|---|
| `VRAM hits` | 2.0 ms | 0.6 ms | the routed experts, GPU 0's misses read from the pinned mirror over PCIe |
| `shared+quant` | 1.7 ms | 1.8 ms | the shared expert (q8_1, 3 GEMVs, SwiGLU, BF16 gate) |
| `hc-read1+router` | 1.5 ms | 1.6 ms | the fused hyper-connection read and the router GEMV + top-10 |
| `hc-read0` | 1.4 ms | 1.3 ms | the fused hyper-connection read |

Over 48 layers those four come to about 190 ms of a ~280 ms window. The excess in `VRAM hits` on GPU 0 is the
mirror, roughly 50 ms per window. Run C shows that borrowing for the prompt path also gives GPU 0 18% more cache
slots (7,761 instead of 6,592), because its prompt buffers no longer take their own VRAM.

**Prompt.** The phase timer (`STRATA_PREFILL_TIMING`) ended the engine at the first prompt: it read event
profiling info from queues created without `enable_profiling`. Patch 15 times the phases with host clocks.

The next profile adds per-kernel GPU time (`unitrace -d`, built by `scripts/get-unitrace.sh`), a real decode baseline,
and two Level Zero runtime switches as A/B runs: the v2 adapter, and the driver's in-order lists.

### Round 2 (04bc8e4)

- **Level Zero runtime switches:** the v2 adapter gave 19.2 tok/s and the driver's in-order lists 19.2, against a
  19.0 baseline. Decode is not limited by kernel submission.
- **Where prompt reading goes:** 90% of it is `dequant`, `gemm gate/up` and `gemm down`, which run per expert. A
  native pack's prompt experts are dequantized to FP16 and multiplied by oneMKL one expert at a time, because the
  grouped MMQ path needs ggml-cuda and is not built in the SYCL port. Each chunk walks all of a GPU's routed
  experts, and on GPU 0 most of them stream from the RAM mirror, so the cost grows with the number of chunks more
  than with the number of tokens. The 256-token first chunk (`STRATA_PREFILL_FIRST`, a B70 latency trick) is a
  whole extra walk here, ~4.7 s.
- **Bigger chunks:** borrowing cache slots for the prompt (`--prefill-borrow --prefill auto`, 8K chunks) took the
  2K prompt from 49 to 116 tok/s and the 8K prompt from 63 to 284 tok/s, with decode unchanged.
- **Defaults now in `./arc start`:** 8K borrowed chunks and no first chunk. Round 3 measures 16K chunks, the
  stream-all walk and per-kernel times (`unitrace` builds without `libze-dev`, since its CMake fetches the Level
  Zero headers itself).

### Round 3 (23394f0)

- **Defaults:** 2K prompt 497 tok/s, 10.7K prompt 589 tok/s (18 s), decode 21.6 tok/s on code and 14.0 on prose.
- **A/B runs:** 16K chunks were slower (437 tok/s): lending 6.7 GiB per card streams more experts. The stream-all walk
  was slower too (363 tok/s). The 8K chunks stay.
- **Prompt phases now:** the experts are no longer the problem. On GPU 0, GDN recurrence takes 31% (3.3 s per 10.7K
  tokens, ~6x the B70's per-token cost) and prompt attention 22%. GPU 1 adds ~4.4 s after its chunks for the MTP
  draft layer's K/V pass, which on its own costs about as much as GPU 1's 13 layers.
- **unitrace:** it was built, but it wrote nothing. The engine leaves through `std::_Exit(0)` (its GPU objects'
  destructors abort against a runtime that is already shutting down), which skips the tool's teardown. With
  `STRATA_EXIT_SIGNAL=10`, the engine raises SIGUSR1 first, and `unitrace --teardown-on-signal 10` writes its
  report from the handler (patch 16).
- **Round 4** sweeps the kernel variants upstream tuned on the B70: SIMD16 GEMVs, the hyper-connection read's
  variants and the GDN recurrence's variants, one run each. It also adds `STRATA_DRAFT_TIMING` and unitrace.

### Round 4 (a40fc8d): the kernel times, and why Alchemist ran the Xe2-tuned kernels so slowly

- **The variant sweep found nothing:** SIMD16 GEMVs, rows per warp and four hyper-connection read variants all
  measured 21.5-21.6 tok/s on code. Of the GDN recurrence's variants, the key-head kernel was 35% slower, and
  the others matched the default.
- **unitrace, decode:** kernel time is about 135 ms of a ~220 ms window. The largest items per call:
  - the hyper-connection up kernel `gr_up_multi`: 269 us, 2 per layer, for 6.5 MB of BF16 weights (~13 us of
    bandwidth)
  - the experts' gate/up GEMV `launch_gu_port<IQ2_S>`: 422 us
  - the experts' down GEMV: 145 us
  - the router's top-10: 91 us
- **The cause, found offline:** IGC's dumps for `acm-g10` (`ocloc -device acm-g10` with
  `IGC_ShaderDumpEnable=1`, then the zeinfo `spill_size` / `private_size` of every kernel) show two separate
  problems. Neither needs the cards to see.
  1. **A 936-byte argument copied per work-item.** `GrMulti`, the per-token arguments of the hyper-connection read,
     was one by-value kernel argument that every work-item copied into private memory and indexed by lane, group
     and token: 29,952 bytes of scratch per SIMD32 thread, ~38 MB of scratch traffic per `gr_up_multi` call. Patch
     17 uploads each distinct `GrMulti` once to device memory (a call site's buffers are the same on every window,
     so graphs replay the same pointer) and the kernels take a reference: `private_size` 0.
  2. **Register spills everywhere.** On Xe2, IGC picks the 256-entry register file per kernel by itself; for
     `acm-g10` it does not, and the auto option changes nothing. At 128 GRFs nearly every decode GEMV spilled:
     `native_mmvq` 10-17 KB per thread, the expert kernels 2.5-10 KB, the hyper-connection read 2-3 KB. Patch 18
     builds the device code with `-ze-opt-large-register-file` on Alchemist (`ARC_LARGE_GRF=OFF ./arc build`
     for the A/B). The decode GEMVs then compile at SIMD32 with no spill: `iq_kernels` goes from spills in nearly
     every kernel to 1 of 150 (224 B), and every `native_mmvq` kernel decode uses to 0. Every kernel stays SIMD32
     or 16, so 1024-item work-groups still fit. `gr_down_sliced` also runs at SIMD16 (its shuffles stay in 4-lane
     groups), which removes its spill at either register size.
- **The draft layer's 2.9 s per chunk:** the serve path batched the drafter's prompt K/V only on one GPU (`!multi_gpu
  && sp.draft_kv(...)`). With a layer split, the drafter read every chunk through its own pass, 6 tokens at a time.
  Patch 19 batches it on the last stage's prompt path, which is on the drafter's device.
- **Prompt trace, next targets:** `gdn_recurrence` takes 36.8 ms per call (8K sequential steps; a chunk-parallel
  form is the fix). Prompt attention runs through the FP32 `qsa_decode_attn_batch` at 724 us x 4,032 calls.

### Round 5 (e819263)

- **Speeds:** decode 25.1 tok/s on code (from 21.5) and 17.6 on prose (from 13.9). The 10.7K prompt reads at 789 tok/s
  (from 589). The batched draft pass takes 128 ms per 8K chunk (from 2.9 s). GDN recurrence 1.05 s (from 3.3 s),
  which the large register file sped up too.
- **Decode kernels now:** about 108 ms of kernel time per ~188 ms window. The rest is gaps between ~2,900 small
  kernels.
  - `gr_up_multi` takes 110 us (from 269).
  - The expert GEMVs lead: `launch_gu_port<IQ2_S>` 368 us per call, but between 3.7 us and 4,350 us. The B70 measured
    52.5 us for it. The spread fits GPU 0's calls reading their misses over PCIe from inside the kernel.
- **Round 6 A/B:**
  - `--vram-reserve-mib 512`, for ~750 more expert slots per card (1.4 GB stayed free after loading).
  - The expert kernels' lanes per row (`STRATA_GU_LANES` / `STRATA_DOWN_LANES`).
  - `--spec 5/6`: windows of 7/8 tokens, since code accepts 93% of the drafts. This needed patch 20: the device plan
    held 64 entries and refused the 80 of an 8-token window, which in a host-free window means a plan nobody sends.

### Round 6 (003ea10): one change per run, against 25.3 / 17.6 tok/s (code / prose)

| run | code | prose | note |
|---|---|---|---|
| `STRATA_GU_LANES=32` | **29.9** | 18.4 | 32 lanes per row in the experts' gate/up GEMV (8 was tuned on Xe2) |
| `--vram-reserve-mib 512` | 27.7 | **19.1** | 1,284 more expert slots across the two cards; 432 MiB left free; 10K prompt 695 vs 757 |
| `STRATA_GU_LANES=16` | 26.9 | 17.7 | |
| `--spec 6` (8-token windows) | 26.4 | 17.5 | |
| `--spec 5` | 25.8 | 17.6 | |
| `STRATA_DOWN_LANES=16` | 25.1 | 16.8 | |
| `STRATA_GU_LANES=4` | 23.8 | 15.9 | |

The lanes change the summation order, so greedy text can differ: prose ran to 128 tokens instead of 107.
`./arc start` now sets 32 gate/up lanes, `--vram-reserve-mib 768` (a margin above 512's 432 MiB free) and
`--spec 6` (lib.sh `arc_engine_env`). Round 7 measures that combination twice per prompt and A/Bs it against
512, `--spec 4`, and 32 / 4 down lanes.

### Round 7 (089cb18): the combination, two requests per prompt

`./arc start`'s defaults (32 gate/up lanes, 768 MiB reserve, `--spec 6`, 8K borrowed chunks, no first chunk):

| prompt | prompt tok/s | decode tok/s |
|---|---|---|
| short code | 42-54 | **34.2** (33.7 / 34.6) |
| short prose (ru) | 55 | 19.4 (18.9 / 19.8) |
| 2K code | 456 / 717 | 26.4 |
| 10.7K code | 814 / **997** | 26.2 |

None of the single changes against it beats noise: reserve 512 34.5 / 19.5 (first 10K prompt 749), `--spec 4`
33.2 / 19.8, 32 or 4 down lanes 33.2 / 18.8 and 33.8 / 19.1, 16 gate/up lanes 31.4 / 18.5. Prose stays below code
because its MTP drafts are accepted far less often (~35 of 100 against ~85).

### Round 8: the misses staged in VRAM, two kernels fixed, any A770 machine in one or two-card mode

- **Patch 21: mirrored experts are copied, not read in place.** The device-built plan put the experts it found
  in the pinned mirror into the VRAM groups with their host address, so the expert GEMVs read them over PCIe a few
  bytes per lane at a time: unitrace's gate/up GEMV ranged from 3.7 us to 4.4 ms per call. The window already has
  the host pool's PCIe share in copy-kernel mode: `fetch_blobs` copies the blobs into VRAM staging with ~100K
  threads of 16-byte loads, then a second grouped launch computes them. `resident_plan` now puts the first
  `pcie_cap` mirrored experts of a group there (`ptr2` / `start2` / `counts[2]`, entries after the VRAM ones: the host
  pool's layout). The cap is the group's share of the staging, 16 (8 per half of a split window). Any experts past
  the cap stay in place, as before.
- **Patch 22, two smaller kernel fixes:**
  - `gr_up_multi` read its staged `lo` rows 32 bytes apart across a sub-group, which causes SLM bank conflicts. They
    are now staged chunk-interleaved and read with a stride: the same products in the same order, so the same bits.
  - The router's 8x32 work-groups, of which rows 1-7 returned at once, are now 1x32.
- **One or two cards (patch 23), any machine:**
  - The runner takes `ARC_GPUS` (indices among the discrete Arcs, the integrated GPU left out) and drops
    `--layer-split` when one card is visible, so a two-card config also runs on one card.
  - `./arc build` derives the AOT target from the PCI ids (`acm-g10/g11/g12`, comma-joined for mixed cards).
  - `./arc doctor` prints a plan per mode for the machine's VRAM and RAM. It is calibrated on the two-card run:
    21 of 32 GB held experts.
  - `./arc profile` adds a one-card run (G1) on two-card machines.

### Round 11 (2026-10-08): prompt crossover after grouped experts

The focused `ARC_PROFILE_SHORT_READ_SWEEP=1 ./arc profile` run
(`profile-20261008-012913.txt`, patch 31, two replies per case) compares complete reply times:

| short-read | agent, 311/314 tokens | code, 730/733 | code, 1425/1428 | code, 2125/2128 |
|---|---|---|---|---|
| 64 | 4.0 / 3.3 s | 7.5 / 7.0 s | 9.4 / 9.3 s | 11.9 / 11.8 s |
| 256 | 4.0 / 3.3 s | 7.5 / 7.0 s | 9.4 / 9.3 s | 11.9 / 11.8 s |
| 512 | 8.0 / 7.9 s | 7.9 / 7.0 s | 9.6 / 9.3 s | 11.9 / 11.8 s |
| 1536 | 8.1 / 8.0 s | 15.5 / 15.3 s | 28.0 / 28.0 s | 12.4 / 11.9 s |

64 and 256 select the same batched path for these cases and tie. The default is now 64. The old 1536 threshold
was fitted to the FP16/oneMKL expert path; it is too high after patch 30.

Patch 32 adds a SYCL prompt-only entry point and `STRATA_PF_EXPERT_KERNEL=dp4a|tile8|tile16`.
The larger DP4A tiles reuse decoded weights across 8/16 activation rows, with the original four-row tail.
The decode entry point keeps its existing kernels. The diagnostic pack uses Q2_0 down in all layers, so this first
profile isolates the IQ2_S gate/up change (34 of 48 layers); the wider IQ4_NL down kernels are not selected here. IGC dumps for acm-g10 with the large register file report
SIMD32, 256 GRFs and no spill/private-memory fields for both gate/up and down variants.

With the timer off, `profile-20261008-020158.txt` measures 10K prompt reading at 363.9/370.2 tok/s (DP4A),
376.2/382.4 (tile8) and 382.5/389.7 (tile16). The first complete reply check
(`check-20261008-015237.txt`, tile8) passes all 14 requests in base, own-buffers and one-gpu.

`tools/build-prefill-parity.sh` builds a focused numerical check from the patched kernel library.
`test-logs/prefill-parity <shard1.gguf>` picks one real expert layer per gate/up format, compares against ggml's
float dequantization and FP64 dot products, and tests 7/17/65/160 entries with shuffled input and output mappings.
Its 3% L1 tolerance is the upstream native-expert test's (activation quantization differs from the float reference).
DP4A, tile8 and tile16 pass with about 1.1-1.2% relative error. This test is correctness-only; timings come from
`./arc profile`.

### Q2_0 down tiles (patch 33)

The diagnostic pack uses Q2_0 down (type 42), not IQ4_NL (type 20). Patch 33 widens this actual down
path as well. `profile-20261008-022225.txt`, two replies per case with phase timing off:

| prompt | DP4A (4 rows) | tile8 | tile16 |
|---|---|---|---|
| 2K prefill, tok/s | 249.9 / 271.8 | 277.6 / 304.3 | 286.6 / 312.9 |
| 10K prefill, tok/s | 363.3 / 370.6 | 409.1 / 419.3 | 426.0 / 433.3 |
| decode after 10K, median tok/s | about 30 | 30.0 | 29.9 |

The tile16 10K median is 17% above four-row DP4A. `check-20261008-021323.txt` passes all 14 cases
in base, own-buffers and one-gpu with tile16. The numerical parity test now also compares each output against
the original DP4A kernel: all three tested gate/up formats and all four tail sizes have zero L1 delta for tile8
and tile16. Decode still uses its original entry point.

### Staging banks and phase timing (patch 34)

Patch 34 adds opt-in STRATA_PF_STAGE_BANKS (2-4), STRATA_PF_STAGE_SLOTS (1-128),
STRATA_PF_GROUPS (8-128) and STRATA_PF_BATCH_ROWS (up to the current chunk).
It also enqueues a batch's copies before the phase timer waits for the previous compute.
This changes phase instrumentation, not the ordinary untraced schedule.

With tile16, profile-20261008-134607.txt measures:

| banks / slots / group cap / row cap | 10K prefill tok/s, two requests |
|---|---|
| 2 / 32 / 64 / chunk | 426.5 / 433.5 |
| 3 / 32 / 64 / chunk | 424.7 / 431.3 |
| 3 / 16 / 64 / chunk | 418.3 / 424.1 |
| 3 / 64 / 64 / chunk | 424.6 / 432.0 |
| 3 / 32 / 32 / chunk | 420.9 / 427.4 |
| 3 / 32 / 64 / 2048 | 382.5 / 390.0 |

None wins: defaults remain two banks, 32 staging slots, 64 groups and the whole chunk.
A-CP2 and A-CP3 both report wait copy at 13.7% for the complete 10K prompt.
The synchronous phase timer still changes overlap, so this is a diagnostic result, not a speed gain.
check-20261008-133701.txt passes all 14 cases in base, own-buffers and one-gpu with three banks.

### Alchemist SG8 expert matrices (patch 35)

The SG8 FP16 8x8x16 shape compiles and runs on ACM-G10. Numerical parity passes for all three
real gate/up formats, shuffled entries and 7/17/65/160-row tails. It is slower than DP4A here:
profile-20261008-142124.txt measures dual-card 10K at 426.9/432.6 tok/s with tile16,
88.4/88.8 with 256 GRFs and 134.4/134.9 with 128 GRFs. Lowering the row threshold to one
still gives 134.3/135.0. On one card profile-20261008-160840.txt measures 108.7/109.1 versus
tile16 369.1/374.4. Keep XMX opt-in; legal matrix shapes alone do not imply a speed gain.

### PCIe probe allocation (patch 36)

The previous probe measured pageable malloc memory. STRATA_PCIE_PINNED_PROBE=1 uses USM host
memory and host elapsed time through memcpy.wait_and_throw. The engine log reports 13.7-13.8 GB/s
on the x16 card and 5.2 GB/s on the x4 card, versus about 9.3 and 4.5 with pageable memory.
These are probe measurements, not effective expert-copy throughput.

profile-20261008-160840.txt, two replies per case:

| probe allocation | dual 10K tok/s | single 10K tok/s |
|---|---|---|
| pageable | 431.2 / 439.5 | 369.1 / 374.4 |
| pinned | 430.6 / 438.5 | 369.1 / 374.5 |

Probe pinning does not speed up prefill. The phase-timed pinned run still has 12.8% wait copy
on the main stage. check-20261008-155935.txt passes all 14 cases in all three configurations.
All seven profile runs finished; a shell tail parse error caused by editing a running script prevented
its final summary. The script passes bash -n, and the summary was recovered from the original saved
bench rows without rerunning or replacing measurements.

The same profile finally compares tile16 against DP4A on one card: 369.1/374.4 versus 312.3/316.2
on 10K, an 18.3% median improvement. Dual improvement is about 17%. Earlier single-card figures
around 310 are not used as a matched baseline for this newer run.

### Pinned copies of lent experts (patch 37)

STRATA_PF_PIN_LOAN=1 copies immutable experts to USM host memory before their cache slots become
prompt scratch. Pools use at most half of MemAvailable, one-GiB allocations, and retain the existing
Stager fallback. These extra pinned copies are opt-in and consume additional RAM.

check-20261008-163459.txt passes all 14 requests in each configuration. In
profile-20261008-164542.txt (two replies, including initial pool preparation):

| mode | dual 10K tok/s | single 10K tok/s |
|---|---|---|
| ordinary sources | 429.9 / 438.6 | 367.8 / 375.0 |
| pinned loans | 411.6 / 451.0 | 369.2 / 382.3 |

The warm dual request is 2.8% faster, but the two-request median is 0.7% slower; the single median
is 1.2% faster. Do not enable this alone by default. Phase timing on the main stage reports
wait copy 12.7% -> 10.7%, gather 313 -> 59 ms, and host staging 950 -> 535 ms for the second
10K request. The requested <5% wait-copy target is not reached. These phase runs include observer
synchronization and are not the final speed measurement.

Inspection found that the later cache refill still uses the original source and drains its temporary
blob ring every 256 entries. A separate experiment will reuse immutable pinned copies there and
keep the ring drains only for fallback blobs. Startup preparation is tested separately from lazy pinning.

### Early pinned expert copies (patch 38)

STRATA_PF_PREFETCH_SLOTS=64/128/256 allocates a separate pool, limited to half the currently
free VRAM, and issues pinned misses while attention runs. It must not use the tail of Xs: Xs
aliases attention scratch. The first implementation considered that alias and was not executed;
the shipped experiment uses disjoint memory. Default is zero.

check-20261008-170142.txt passes all 14 cases in base, own-buffers and one-gpu.
profile-20261008-171042.txt measures:

| slots | dual 2K tok/s | dual 10K tok/s | single 2K tok/s | single 10K tok/s |
|---|---|---|---|---|
| 0 | 286.8 / 316.5 | 431.5 / 439.1 | 290.2 / 310.3 | 369.3 / 374.7 |
| 64 | 286.9 / 316.6 | 432.4 / 441.6 | not measured | not measured |
| 128 | 284.3 / 312.7 | 433.4 / 439.3 | 290.5 / 307.4 | 372.2 / 377.4 |
| 256 | 280.4 / 306.3 | 433.6 / 439.8 | not measured | not measured |

The dual 10K difference is below 0.5%; 128 reduces dual 2K by about 1.0%, and 256 by 2.8%.
Single 128 gains 0.8% on 10K but loses 0.4% on 2K. This does not justify a default change.
The current prefix-of-misses selection is not a routing-frequency predictor. It was tested with
ordinary sources; combined tests with fully pinned loans may change its usefulness.
The timed EP128 run has 9.6% wait copy on the main stage, still above the requested 5% target.

### Host kernel failure during pipeline validation

The first patch-39 check passed base and own-buffers (14 replies each), but the host
became unreachable while loading one-gpu. EFI pstore survives the manual reboot:
Linux 7.0.0-38 had already reported `BUG: Bad page state` / `compound_head not consistent`
at 16:20 in i915 shmem allocation and cgroup reclaim, before patch 39 was applied.
It later faulted in `lru_gen_del_folio` from the same allocation/reclaim path. The
final console stack is process exit/TLB flushing. This is not a BROKEN-reply result
and does not establish whether the original cause is kernel/driver or hardware.

For the resumed experiments only, MGLRU was disabled through its runtime sysfs
switch, and the test unit's explicit 44-GiB MemoryMax was removed. These changes
are not portable engine defaults or a confirmed kernel fix. All subsequent A/B
arms run under the same host settings; a final matched control is required rather
than comparing across the reboot. No kernel/driver/firmware upgrade was performed.
The complete previous-boot journal and EFI pstore were preserved separately.

### Existing stage pipeline control (patch 39)

The upstream prompt path already double-buffers hand-off rows and overlaps its two
stages using std::async. Patch 39 adds STRATA_PF_PIPELINE=0 to wait immediately;
mode 1 preserves existing behavior. MTP, PLE and checkpoint callbacks are retained.
check-20261008-174420.txt passes all 42 requests after the host reboot.

profile-20261008-181213.txt measures dual 10K 362.0/362.6 tok/s serial versus
437.7/437.8 pipelined (+20.8% median); 4096-token chunks give 432.2/448.9.
Single gives 373.7/374.0 versus 373.9/374.3; the switch has no single-stage effect.
This exposes an existing benefit; it is not a newly obtained +21% above tile16.

The normal profile's 32-token greedy comparison failed both across dual modes and
across single modes. Independent fresh-request controls in
profile-20261008-182245.txt compare serial repeat, pipeline repeat and serial versus
pipeline: all 32 tokens and top-20 normalized log probabilities are identical
(max delta 0). The preceding one-token repeat test also has delta 0. However,
32-token single-card repeats still differ with identical mode settings. This
baseline reproducibility limitation is unresolved; passing BROKEN-reply checks
does not establish bitwise deterministic generation across process lifetimes.
Keep the failed comparison files rather than calling those profiles entirely clean.
Default pipeline remains 1; no new stage scheduling is enabled by this patch.

### SG8 prompt attention (patch 40)

STRATA_PROMPT_ATTN_XMX=16/32 enables legal Alchemist SG8 FP16 8x8x16 matrices
with FP32 accumulators, Q high/low halves and register-held output accumulators.
Default remains zero. tools/attention_parity.cpp tests int8 and FP16 KV against
FP64, the FP32 kernel, shuffled page tables, 32768/1500/2100 contexts and the
identity-to-sparse selection edge. Both chunk sizes pass their numerical bounds.

The first mixed int8-K/Q4-V prototype failed the strict bound: max error from
FP64 0.00184 versus FP32 4.15e-6 (about 2.6e-4 of output scale relative to FP32).
Q4 products rounded to one FP16 V tile do not retain enough precision. The patch
therefore refuses both Q4 keys and Q4 values; the caller retains its FP32 path.
The parity tool separately verifies both Q4 fallbacks rather than pretending
that those formats are accelerated. The failed prototype log is preserved.

On 256-query synthetic int8/FP16 inputs, 16-cell SG8 takes roughly 16 ms versus
8-10 ms for FP32; 32-cell SG8 is also slower on the short-context arms. These are
kernel microbenchmarks, not final model throughput. check-20261008-184712.txt passes all 42 replies with SG8 enabled. The completed
profile-20261008-185610.txt gives:

| attention mode | dual 2K tok/s | dual 10K tok/s | single 2K tok/s | single 10K tok/s |
|---|---|---|---|---|
| FP32 | 283.9 / 316.5 | 429.8 / 439.8 | 275.2 / 308.5 | 366.5 / 374.5 |
| SG8 CH16 | 266.5 / 298.7 | 384.2 / 392.9 | 272.8 / 292.9 | 326.1 / 331.3 |
| SG8 CH32 | 278.5 / 309.5 | 412.5 / 421.9 | 287.5 / 303.9 | 351.6 / 356.7 |

CH16 loses about 10.6% dual and 11.3% single on 10K; CH32 loses 4.0% and 4.4%.
The small single 2K gain for CH32 does not offset long-prompt regression or the
dual 2K loss. Keep the FP32 default. Legal XMX shapes and precise arithmetic
are not sufficient for faster attention on Alchemist.

## What was verified here, and what was not

| Check | Status |
|---|---|
| Host + device compile of all 84 SYCL TUs at `a5682fe` + patches | done (icpx 2026.1.1) |
| FP64 scan of the device code, `STRATA_SYCL_NO_FP64=ON` | 0 instructions |
| Every per-kernel SPIR-V module through `ocloc -device acm-g10` | 994 of 1001 before patch 6; the 7 failures are exactly the `joint_matrix` modules |
| Host link of the engine | fails at `a5682fe` (#871 drift); links with patch 5 |
| AOT build of `strata` for `acm-g10` with all 6 patches | **links**: 994 device modules built by IGC for acm-g10, 0 compiler errors (the 7 XMX modules are stubs now), 16 min on 4 cores; the 102 MB binary runs (`--help`) |
| A tiny double kernel: `acm-g10` fails, `bmg-g21` builds | reproduced |
| Run on A770 | measured on one and two A770s with full reply checks and profiles; see rounds above |
| Native prefill numeric parity on A770 | real GGUF experts tested against FP64/ggml and the original DP4A kernel; this does not claim that the entire upstream ctest suite passed |

### Round 41: split-format gate/up row tiles

`STRATA_PF_SPLIT_ROWS=4/8/16/auto` extends row reuse to native IQ2_XXS and IQ1_M (IQ2_S already had tile16). Real GGUF parity with shuffled mappings and row tails passes every mode; full check-20261008-191643.txt passes all 42 requests. Profile-20261008-192536.txt (128 output tokens, two runs) gives mean 10K prefill: dual 434.3 / 434.9 / 433.1 / 434.3 tok/s; single 370.9 / 371.2 / 370.4 / 371.8. This is noise, not a reason to change the default of 4. Dual 2K 308.4 / 311.4 / 308.9 also has no robust gain.

### Screening policy for remaining candidates

At the user's request, test potential speed gains before running the full stability matrix again. `ARC_PROFILE_SCREEN=1 ./arc profile` preserves two cold/warm measurements, uses 2K/10K with 32 output tokens for prefill candidates, and keeps normal lengths for decode/spec/short-read and parity diagnostics. Profiles record screening, cases, output length and run count explicitly; their short-output decode rates are not final throughput figures. Fusion retains real-weight numerical parity. A candidate losing over 5% in every measured dual-GPU prefill case skips redundant single-GPU screening (unmeasured single performance is not claimed). Only measured winners go through the full final check and normal-length profile. Host-copy rounds 43/44/45 remain separate patches but share one build; all experimental defaults stay off until justified.

Screening also prunes SR256 in the card sweep: every measured agent/0.7K/2K prompt is at least 311 tokens, so thresholds 64 and 256 select exactly the same batched path. SR768 remains a distinct comparison. Lazy-loan refill sweep is omitted in favor of the matched eager-loan P2R0/P2R1 comparison, which isolates the same refill switch without extra server reloads. Unitrace graph/eager samples use short code and 64 generated tokens; normal decode throughput still uses code/prose with 128 tokens.

### Round 42: fused native expert, rejected

`STRATA_PF_FUSE_EXPERT=4/8/16` computes gate/up, SwiGLU, q8_1 hidden quantization and Q2_0 down in one work-group, using local hidden buffers and guarded row tails. Build succeeds; every mode passes real-GGUF FP64-reference/DP4A parity on IQ2_XXS/IQ2_S/IQ1_M, shuffled mappings and 7/17/65/160 rows, DP4A L1 difference about 3e-8. This does not make it faster. Screening profile-20261008-194950.txt (two runs, 32 output tokens): baseline means 2K302.1 / 10K434.0 tok/s; fuse4 134.3 / 272.1; fuse8 100.5 / 185.3. Fuse16 measured 2K23.0/23.4 versus baseline302.1 (over92% slower); its long tests were explicitly aborted under the user's fast-screening instruction. 10K fuse16 and all fused single-GPU speed figures are unmeasured, not inferred. The profile is interrupted and is not labelled a successful complete profile. All fused paths remain opt-in; default0 retains separate kernels. Larger local buffers and per-group work trade off occupancy/reuse; these measurements reject this design without claiming a verified low-level bottleneck.

### Rounds 43/44: queued immutable refill and eager pinned loans

Round43 `STRATA_PF_PIN_REFILL=1` restores borrowed cache slots directly from immutable pinned loan copies; the 256-buffer reuse drain is retained only for fallback pageable staging. Round44 `STRATA_PF_PIN_LOAN=2` prepares those bounded copies before READY, instead of doing it lazily in the first prompt (mode1). Both default off pending final validation. Budget and fallback are runtime-based; no PCI address, GPU count or model path hardcode is introduced.

Screening profile-20261008-201005.txt, two runs with 32 output tokens: dual P0 2K301.1/10K433.9; P2R0 273.4/417.6; P2R1 329.4/473.5 tok/s. Single P0 298.1/370.9; P2R0 298.2/373.4; P2R1 320.55/391.6. Combined P2R1 is a real candidate (+9.1% dual10K, +5.6% single10K; +9.4%/+7.5%2K), not yet a validated default. Eager preparation alone is insufficient. The phase observer still reports 10.7% wait-copy on the primary stage, so the <5% target remains unmet; cache-restoration waits also occur outside this phase interval. Final normal-length matched controls and full stability matrix are required before adoption. Chunk/short-read/GDN/decode sweeps now use this candidate, since it changes the cost of the batched path. The single-card third staging bank regresses slightly (profile-20261008-202122.txt:2K319.6→312.7,10K391.6→389.9); keep2.

One interaction probe is warranted: eager pinned copies make borrowed experts eligible for early prefetch, whereas the earlier round38 test with PIN0 could only prefetch the existing pinned mirror. `ARC_PROFILE_EARLY_COMBO=1` compares0/128 slots on P2R1; single-GPU arms run only if dual10K improves at least3% without a clear2K loss. No new kernel or rebuild is required.

### E/F/H screening after the copy candidate

Profile-20261008-202730.txt (P2R1, two10K runs,32 output tokens): dual GDN base467.5,keyhead357.8,heads462.8,serial465.6 tok/s. The keyhead variant loses23.5% and its single-GPU repetition is skipped. None justifies a new default. HC read is ~3.2% of the primary-stage timeline; retain its current path rather than add an unmeasured precision change. Profile-20261008-202122.txt tests two/three staging banks on one GPU; no third-bank benefit as recorded above.

CARD screening uses normal128 output tokens for agent/short-read/spec; all knobs are measured on P2R1. Initial dual SR64 wall: agent3.94/3.03s,0.7K6.84/6.48s,2K10.49/10.19s. SR768:agent8.01/7.70s,0.7K15.33/15.06s,2K10.74/10.33s. Keep64; reducing batched overhead strengthens this choice.

### Reboot at 21:28 and completed late screening

The second host loss has a preserved previous-boot journal: at21:12, Linux7.0.0-38 reports list_del corruption in kswapd/isolate_lru_folios. The incorrect prev->next differs in one bit from the expected pointer. This is evidence of memory corruption, not proof of a DRAM fault or an engine cause. Disabling MGLRU did not prevent it (the second stack is classic LRU). No MCE or OOM diagnosis is established; BIOS remains1.08. The watcher now also stops on list corruption, which the old BUG/Oops pattern missed.

Profile210736 rejects early128 on P2R1: dual10K mean473.85->468.3 and2K328.35->319.9; single repetition skipped. Profile210039 normal128 decode with graph: dualcode33.9/34.8,prose19.1/19.7;singlecode25.1/25.7,prose14.0/14.7. Traced64-token eager runs regress to10.5-11.8tok/s. Keep graph; these traced figures are not normal throughput. Single32-token eager repeats in profile210532 still differ, so disabling graph does not resolve the generation reproducibility limitation.

DMA profile211052 was interrupted after dual arms and before the single arms. Real event clocks on P2R1 show primary pinned-loan15.25-15.29GB/s and mirror15.11-15.12GB/s;later-card pinned-loan4.93-5.16GB/s. Primary wait remains10.6-10.9%, not below5%. Do not claim25GB/s or completed single-copy clocks.

CARD203752 contains failed2K own-buffer2048 requests (HTTP400/503), with user-space SIGSEGV in libur_loader on both card configurations. Both are rejected; ordinary measured10K also regresses. The old bench masked API errors if other cases succeeded. tools/bench.py now records failed rows and returns3 for any API/transport error; a synthetic failure+success regression confirms this. Error-body handling excludes authentication bodies and redacts keys. The old CARD file is marked incomplete, not a clean stability result. Single spec6 code25.2/25.7 versus spec8 24.2/23.7 confirms keeping6 for code; prose remains13.9-14.6, so35-40tok/s cannot be claimed for every output type.

### Follow-up goals: throughput and 128K

The current model shard metadata reports qwen4exp.context_length262144 and RoPE base10000000;128K is within that declared limit. This does not establish memory fit or actual long-context quality in this port. Added a counted120K-126K request with audit-label retrieval and a >131072-token overflow control, plus normal API/agent benchmarks. Context expansion is kept separate from32K measured defaults until verified. Layer-placement screening compares auto with explicit34/28/24 and foreign dense-weight trimming, preserving precision, copy flags and normal128 output length. No1000tok/s result has been measured.

### Validated copy default at32K

check-20261008-213531.txt passes all42 requests (base,own-buffers,one-gpu), including both10K rows per configuration. Normal128-output profile-20261008-214426.txt completes all arms with the corrected bench. P0/R0 control versus P2/R1 means: dual10K435.95->474.15 (+8.8%),2K312.65->337.2 (+7.9%);single10K371.7->390.85 (+5.2%),2K309.1->323.1 (+4.5%). Dual0.7K242.75->245.3 is noise;single210.75->210.65 unchanged. Defaults now choose PIN_LOAN2/PIN_REFILL1 with bounded allocation/fallback. Extra pinned RAM is roughly6.9GiB dual /3.4GiB single for the current loans; this is not free memory savings.

Actual normal decode after10K is dual29.6 versus31.1 control andsingle22.35 versus22.45; the two-run dual difference is preserved, not called an improvement. Short code is33.9/34.8 dual and25.2/25.7 single;Russianprose18.7/19.2 and14.0/14.7. Agent tool stream38.3/38.8 dual,26.2/26.6 single has only26 generated tokens. These figures must not be generalized to stable35-40tok/s after every prompt.

The residency knob UR_L0_USM_RESIDENT is supported by the installed legacy adapter but absent from the installed v2 library strings;process maps confirmv2. Intel's adapter-specific documentation also lists the equivalent legacy knob asLegacy-only (https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md). No inert residency variable was made a default, and no reliability fix is claimed.

### Stage placement tradeoffs

Normal128-output profile215453, two runs per case, P2R1: auto34/14 means10K473.5,short-code33.45,after10K31.35,Russianprose19.25tok/s. Explicit34+trim gives474.6/35.9/29.4/19.75;28+trim446.9/37.6/33.15/18.1;24+trim420.9/34.6/27.5/16.0. These are tradeoffs, not a general faster prefill. None meets the no-clear-regression selector: keep auto for the128K capacity probe. A coding-focused user may prefer28+trim, but its~5.6%long-prefill and~6%prose losses must be visible. The1:1 placement does not help on the x16/x4 pair. Temperatures reached84C upper/72C lower in the sampled sweep without recorded thermal-throttle flags; this does not prove cooling can be ignored indefinitely.

### Host RAM integrity failure blocks further stability claims

Before the128K capacity probe, CPU-only memtester4.6.0 tested4GiB of successfully mlocked memory with no Strata engine running. It completed with exit4 and8 FAILURE comparisons across SolidBits,BlockSequential,Checkerboard and later patterns. Seven show XOR0x0004000000000000 (bit50), the same bit as the21:12 corrupted kernel pointer;one shows bit4. Logs are retained under test-logs and copied separately. This is direct evidence of host memory integrity failure independent of active model inference, but userspace offsets do not identify a DIMM or distinguish DRAM/controller/board. The machine has two8GB Kingston and two16GB A-DATA modules, all reported at2400MT/s, no ECC. A shorter fresh CPU confirmation was launched with SolidBits/Checkerboard only.

The shell chain stopped on memtester's failure, before creating a128K config or issuing128K inference. No128K capacity, quality, throughput or editor compatibility result is claimed. GPU/server tests are left stopped pending repair of host memory integrity;8790 was not started. All earlier speed tables describe observed measurements, not long-term host stability or a proven performance ceiling. New1024+prefill designs should be screened only after the host passes memory tests.

The fresh4GiB confirmation using only SolidBits/Checkerboard (MEMTESTER_TEST_MASK0x500) completed exit0. This narrower pass does not clear the earlier8 failures or test all48GiB. Full offline per-module testing is needed; no module is declared faulty solely from these offsets.


## Follow-up after the 2026-10-08 reboot

The operator restored base memory clocks and requested no further RAM tests.
Optimization resumed with kernel-log observation only. Historical failures above
remain historical evidence; no new RAM certification is implied.

`profile-20261008-222222.txt` and its C128 context JSON passed the three-label
124K retrieval, long streaming tool call, and >131072 input rejection checks.
This configuration uses all-VRAM int8 KV and automatic device placement. Its long
request achieved 525.1 prompt tok/s, with only nine answer tokens; the 19.3 decode
rate on that very short answer is not a steady-state decode benchmark.

Patch 46 is a separate opt-in integer XMX experiment: legal SG8 8x8x32 signed
int8 multiply/int32 accumulate, with original IQ2_XXS integer post-dot scaling and
Q2_0 signed codes, converted to FP32 at each 32-element activation block. No
full dequantized weight matrix is allocated. Cached per-thread device queries
check matrix type, shape and SG8 before dispatch. Other formats and decode keep
the existing path. Numeric and performance results are recorded below once run.


The first integer XMX screen (`profile-20261008-223950.txt`, 32 output tokens,
two runs) regressed: 10K means DP4A 472.05, integer GU 255.15, integer down 211.4
prompt tok/s. No single-card repeats were spent on those clear losses. Patch 47
adds 256-element K slabs and register-only coordinate-preserving accumulation;
its `profile-20261008-230026.txt` still lost (GU about 262, down 215 prompt tok/s).
All 48 slab32/slab256 numeric cases passed. Patch 48's padding/small-group DP4A
fallback is opt-in; only numeric validation is scheduled, with full performance
screening deferred after the repeated XMX losses.

`profile-20261008-224701.txt` multilingual means (256 output cap, two runs):
| mode | code | Russian | English | Chinese |
| --- | ---: | ---: | ---: | ---: |
| spec2 | 22.75 | 21.45 | 25.70 | 23.60 |
| spec6 | 32.00 | 19.90 | 25.80 | 23.80 |
| spec6, confidence0.35 | 31.70 | 18.45 | 23.70 | 22.65 |
| spec6, confidence0.65 | 32.55 | 20.90 | 25.65 | 24.30 |
The optional spec0 arm failed because native packs require spec>=2, and remains
explicitly marked failed in that report; future sweeps omit it. Supported arms
completed without HTTP/BROKEN errors. No confidence default was changed for the
small/variable gains.

`profile-20261008-230726.txt` passed the same real 128K tests with KV resident32768
and automatic placement: retrieval 123888 tokens at 527.9 prompt tok/s, streaming
tool request 124184 tokens at 529.8. Normal 10K mean 468.55 prompt /29.8 decode,
short code33.15, short Russian18.9. Relative to the all-VRAM128K run, measured
cases show a small decode improvement without a prefill loss. Tool decode35.1
on only26 output tokens is not a steady-state35–40 guarantee.

Dispatch inspection found the old PF_SPLIT_ROWS sweep did not force the split
kernel: g_split_multi was false (STRATA_EXPERT_SPLIT was verified unset in the
live engine and runtime config), and launch_gu returned the port kernel before
the multi-column implementation. Multi<16> and Multi<29> are absent, so the port
path falls back to per-entry dots for these formats. The old4/8/16 sweep cannot
be treated as a speed comparison of actual decoded-weight reuse. Patch49 adds a
prompt-only STRATA_PF_GU_SPLIT switch and explicit force_split argument; it uses
the existing Split traits and leaves decode dispatch unchanged. New numeric and
matched speed measurements are required before changing its default.

### Prompt decoded-weight reuse (patches 49–50)

The earlier NC4/NC8/NC16 sweep did not actually select `Split<TG>` for formats 16 and 29: `STRATA_EXPERT_SPLIT` was off, and `launch_gu` returned the port fallback before reaching the multi-column path. Those results cannot establish the performance of weight reuse. Patch 49 adds a prompt-only force flag; decode stays on its original dispatch. No persistent unpacked weight matrix is allocated.

After reboot at the user's baseline RAM settings (no RAM test), `profile-20261008-232414` gives dual-card 10K means: control 472.0, NC4 512.3, NC8 519.6, NC16 518.7 tok/s. All three also improve single-card prefill. Select NC4: within 2% of the best common score, with fewer live accumulators. The 36 focused numeric cases match the decode DP4A result exactly; `check-20261008-234748` passes all 42 cases across base, own-buffers and one-GPU, including long prompts and tool streams.

Patch 50 makes 8/16 lanes per row available only for the new prompt reuse path. Its 72 focused numeric cases pass, but `profile-20261008-234313` finds no >=3% dual-card improvement, so the 32-lane path remains selected. It is an opt-in experiment, not a default speed claim.

Normal-length confirmation (`profile-20261008-235624`, two runs, 128-token cap):

| Mode | 2K prefill | 10K prefill | Short code decode | RU prose decode | 10K code decode | Agent decode |
|---|---:|---:|---:|---:|---:|---:|
| Two GPUs | 360.9 | 511.8 | 33.6 | 19.1 | 30.2 | 37.4 |
| One GPU | 342.9 | 423.2 | 24.8 | 14.1 | 22.0 | 25.9 |

This demonstrates a prompt throughput gain, not 1000 prefill or universal 35–40 decode. Agent replies are shorter than the 128-token cap; language and draft acceptance still affect decode. Launcher defaults now select `STRATA_PF_GU_SPLIT=1` and `STRATA_PF_SPLIT_ROWS=4`; explicit environment overrides remain effective. Context size and layer placement remain user-/hardware-selected.

128K streaming after the GU change (`profile-20261009-000204`): the actual 124209-token request processes fresh input at **593.5 tok/s** and retrieves all three distributed labels into a streaming tool call; oversized input is rejected with HTTP400. A separate 256-token code generation at 124219 tokens decodes at **25.0 tok/s**, not 35–40. It reuses 114688 tokens and reads 9531; do not count reused tokens as fresh prefill throughput. Both requests are saved in the context JSON.


### Draft vocabulary coverage

`profile-20261009-002531` compares CJK (106299 IDs), CJK/Cyrillic union
(124737 IDs) and full classifier with two 256-token runs per language.

| Draft subset | Code decode | Russian decode | English decode | Chinese decode |
|---|---:|---:|---:|---:|
| CJK | 32.10 | 19.95 | 25.85 | 23.85 |
| Union | 31.70 | 29.85 | 25.65 | 23.60 |
| Full | 31.45 | 29.55 | 25.05 | 23.20 |

All 24 cases pass. The union's Russian gain is about 50%; other measured
languages lose about 1%. This does not establish universal 35–40 tok/s.
`tools/prepare-draft-vocab.py` creates separate MTP views with shared dense/expert
files, so the existing runtime subset is not overwritten.
Single-card confirmation (`profile-20261009-003400`, 16 clean cases): code
23.65 -> 23.60, Russian 14.50 -> 21.85, English 20.10 -> 20.05, Chinese
16.85 -> 16.80 tok/s. The same relative language-specific gain holds on one GPU.

The isolated union also passes 12 live checks, two sampled/agent runs and a
123928-token distributed retrieval/tool request (`test-logs/context-multilingual.json`).
Its separate 256-token code probe at 124219 context tokens decodes at 24.5 tok/s;
114688 cached tokens are not fresh prefill. The fresh capacity probe ran while
a model download was active and is a correctness result, not a matched speed claim.

Power screening on the measured host: setting 180/160 W sustained limits leaves
10K prefill around 510 tok/s and code decode around 32 tok/s, but does not prove
cooling. The driver averages over 28 seconds; measured mean GPU power is below
160 W and short peaks still reach ~187 W. Warm-up explains increasing peaks
from 81 to 85 C across the arms. Both 225 W limits were restored. Sysman reports
voltage-offset control unsupported and fan control unavailable; no undervolt or
fan curve has been applied. Power limits are host experiments, not repo defaults.


### Alternative Q2_0 quantization screen

The Q2_0 first shard was downloaded and SHA256-verified; its identical second
shard was reused. Same engine, 128K/32768 resident KV, MTP union, copy settings
and host power limits as the IQ2_XS serving configuration. Two 256-token runs
per case (`test-logs/bench-q2.json`): 2K prefill 287.8, 10K prefill 401.9; code
decode 26.5, Russian 23.1, English 21.95, Chinese 18.2, 10K code 24.0 tok/s.
All 16 benchmark responses and all 12 short live checks pass. It loses to the
measured IQ2_XS configuration on both prompt and decode throughput, so it is
not promoted; no full Q2_0 profile or 128K stress repeat was run. These results
are specific to this engine/hardware, not a general ranking of quant formats.

Matched 128K union control (`test-logs/bench-iq2-trim.json`) confirms IQ2_XS
10K prefill 507.7/code decode 29.9, short code 31.6, RU 29.55, EN 25.45, ZH
23.35 tok/s. `STRATA_STAGE_TRIM=1` alone was inactive with automatic splitting:
the engine condition requires an explicit split. Treat this run as a control,
not evidence of trimming performance.

An explicit 34/14 split (equal to the control's chosen split) activates trimming
and is confirmed by the dense-layer loading lines. Two 256-token runs per case:

| Case | Auto control decode | Explicit trim decode | Explicit trim prefill |
|---|---:|---:|---:|
| Short code | 31.60 | 32.90 | 47.40 |
| Russian | 29.55 | 30.35 | 58.20 |
| English | 25.45 | 26.10 | 54.50 |
| Chinese | 23.35 | 24.05 | 52.80 |
| 2K code | 29.05 | 30.30 | 357.15 |
| 10K code | 29.90 | 31.15 | 510.00 |
| Sampled code | 28.35 | 31.00 | 53.65 |
| Agent | 40.00 | 43.20 | 146.20 |

All 16 responses pass. Agent replies are short; their rate is not a steady-state
43 tok/s guarantee. This split is an opt-in host result, not a common default.
Final validation passes all 14 live checks, including both 10K rows, and the
124209-token retrieval/tool request (`test-logs/context-final.json`): fresh
prefill **597.0 tok/s**, zero reused input. A separate 124219-token code request
produces 256 tokens at **25.5 tok/s**, with 114688 cached input tokens and 9531
fresh tokens. Overflow is rejected. This establishes the tested capacity and
correctness cases, not universal long-context quality or long-term stability.


### Prompt expert parallelism (patch 51, experimental and off)

`STRATA_PF_EP=1` keeps attention/GDN/router/shared experts on the layer owner,
then runs disjoint routed experts on both cards with the existing grouped
kernels. Decode placement is unchanged. The peer owns a bounded double-buffered
16-expert cache; this does not assume that decode's cache holds every layer on
both devices. Activation and result transport use pinned host memory, not P2P.
Results remain FP32 to retain arithmetic parity. An actual pinned PCIe probe
sets the peer row share: approximately 28.5% from x16 to x4 and 65% from x4 to
x16 on the measured A770 host. Single-card mode never activates this path.

The first version serialized peer upload/compute/download and disabled the
existing stage pipeline: 10K prefill fell from 512.25 to 354.0 tok/s. The revised
version overlaps two weight/result banks and retains the stage pipeline
(`STRATA_PF_EP_PIPELINE=0` is the serialized diagnostic fallback). It improves
that experiment, but still loses to the original prompt path:

| Cards/case | EP off | EP on | Unit |
|---|---:|---:|---|
| Two, 2K code | 370.25 | 304.10 | fresh prefill tok/s |
| Two, 10K code | 512.35 | 423.00 | fresh prefill tok/s |
| One, 2K code | 335.45 | 335.00 | fresh prefill tok/s |
| One, 10K code | 421.70 | 421.25 | fresh prefill tok/s |

`profile-20261009-032344.txt`: two normal 128-output-token runs per case,
128K context with 32K resident INT8 KV, multilingual MTP union, explicit 34/14
split on two cards; the paired runs differ only in EP. The EP-on phase observer
is separate from the speed comparison. Its 10K owner timeline has 56.4% expert
compute, 11.5% attention and 4.7% wait-copy; it excludes peer work that overlaps
the owner, so those percentages do not explain total wall time by themselves.

`check-20261009-031417.txt` passes all 42 requests (base, own buffers, one card;
14 each including both 10K requests). The first 32 greedy tokens and all top-20
normalized logprobs match exactly (maximum delta 0). Temperature telemetry peaks
at 83 C / 62 C with stock settings. These tests establish the measured correctness
cases, not an EP throughput win. `STRATA_PF_EP` remains **0** by default; the
800/450 tok/s targets are not met by this change.


### Decode trace and sequencing controls (patches 52–53)

The existing verifier already captures one graph per window size per card.
`NO_HOST=auto` resolves to on for this cache/mirror coverage. Untraced DT in
`profile-20261009-033357.txt` measures short code at 138.92 ms/window:
125.90 verify, 2.59 commit/emit and 10.42 MTP; 4.74 output tokens/window.
Per-layer host waiting and planning are zero. Long Russian prose is 80.18
ms/window and post-10K code 112.83 ms/window, with different window sizes and
acceptance rates. The placement log's CUDA-calibrated "predicted ... per decode
window" is a heuristic for choosing the split, not a measurement to subtract
from real window time.

Unitrace's 48-layer intervals run from the first resident-plan kernel through
the main head sampler (prefix work excluded). Seventeen T=6 graph windows
average 133.35 ms of span, 115.45 ms summed/union kernel time and 116.04 ms
including copy/reset/barrier operations. Eager's corresponding intervals are
336.29/135.19 ms. These traced intervals include observer overhead; DT supplies
normal throughput. There is an approximately 18-ms internal gap in this graph
sample, not evidence of 100 ms spent waiting on the host.
`tools/analyze-decode-trace.py` streams the Chrome timeline without loading the
full 200+ MiB trace into a JSON object. Unittrace inserts a PID before its output
extension; profile.sh now collects those filenames and their interval analysis.

Both A770 devices report `usm_atomic_host_allocations=false` and
`usm_atomic_shared_allocations=false` in the installed oneAPI runtime. A system
atomic flag shared by the two GPU contexts is therefore not a supported SYCL
handoff primitive here. No unbounded device spin or unsupported system atomic
handoff was introduced. The existing explicit host-buffer transfer remains.

Patch 52 adds opt-in `STRATA_VERIFY_COMMIT_OVERLAP=1` (independent stage commits
and MTP overlap, drained before another window/checkpoint),
`STRATA_MTP_CHAIN_SUBMIT=1` (fixed min_p=0 chain queued with one final host wait),
`STRATA_VERIFY_SKIP_EMPTY=1` (remove the unused post-launch queue-empty query),
and `STRATA_VERIFY_NATIVE_GRAPH=1` (configure native recording before SYCL starts;
0 restores the previous recorder). Positive min_p retains the existing early
stop. The original graph path remains the default pending the speed gates.

Patch 53 adds `STRATA_MTP_LOOKAHEAD=1`: a separate MTP queue/thread on the last
GPU continues the previous approximate draft residual while GPU0 verifies.
Its proposals can be used only after full target acceptance and an exact
next-head-token match. Accepted MTP KV is recomputed from actual verifier
residuals/tokens before continuing. Another lookahead is suppressed until a
normal MTP round rebases the chain. Target state and sampled output are never
replaced by an approximate residual. Single-card, sampled, penalized, confidence
threshold and suffix-lookup requests retain the original path. The MTP ring's
existing extra 4*max_t+64 cells preserve the causal rollback window. Eligibility
checks neutral penalty parameters, not whether a history buffer exists (the
server keeps history even without active penalties). Launch/reuse counters
establish whether the path was actually exercised.

All new controls remain zero until correctness and normal dual/single speed
comparisons complete; results below will record the measured outcome.

### Decode experiment results before the next boot

`profile-20261009-041626.txt` was interrupted at the user's pause request.
Its completed dual-card arms use two normal 128-token runs, with observers off:

| arm | code decode tok/s | Russian prose | after 10K |
|---|---:|---:|---:|
| DC: original min_p=0.5 | 34.65 | 31.05 | 30.80 |
| D0: matched min_p=0 | 34.25 | 25.00 | 29.65 |
| DQ: commit + chain + skip | 34.85 | 25.35 | 30.15 |
| DL: DQ + lookahead | 33.10 | 25.10 | 29.75 |

Lookahead was actually exercised: code launched 20 and reused 8 rounds;
Russian prose launched 43–49 and reused one. It did not improve throughput.
All new defaults remain zero. These are completed arms of a partial profile,
not a completed dual/single validation. The later native-recording arm failed;
`ARC_PROFILE_NATIVE_GRAPH=1` is now an explicit opt-in rather than automatically
repeating that failed mode in every decode sweep. Resume checks reject rows
with errors/null metrics. A failed 10K parity request stops its arm before the
remaining speed requests. The check's own-buffer arm preserves the caller's
extra flags so min_p=0 experiments exercise the intended path there too.

### Data-integrity diagnostics after a reboot

A subsequent boot broke even the saved pre-51 executable with all new controls
off: 10K prefill produced NaN. Checksums of both complete GGUF shards matched
the stored download manifest; the prepared dense pack matched its original
copy. The first NaN was at QSA layer 3's output projection, preceded by finite
but huge residual/projection values (~1e18–1e20). This is not a usable speed
baseline and those failed/partial profile metrics must not be promoted.

Patches 54–60 add opt-in diagnostics and startup-only transfer alternatives:
`STRATA_DBG_PF_PHASES=1` synchronizes/reads intermediate values in layers 0–3;
`STRATA_DBG_WEIGHT_COPY=1` checks dense/native upload bytes, refuses a mismatch,
and repeats its failing window through pinned readback. It never reads unused
MoE dequant scratch as evidence about grouped expert weights.
`STRATA_NATIVE_UPLOAD_PINNED=1` stages native GGUF payloads through one bounded
8-MiB host-USM buffer. `STRATA_DENSE_UPLOAD_KERNEL=1` replaces dense uploads
with a byte-copy kernel reading host USM, automatically staging native payloads.
Both default to zero. Diagnostics are not performance measurements.

On the upper A770, actual weight readback repeatedly differed by bit 0x20
(e.g. source 76/readback108, 8/40, 70/102, 196/228). Pinned repeats confirmed the
same byte; native CPU staging matched the file before and after upload. Forcing
the v1 adapter, disabling its USM pool and a copy kernel did not remove the
failure. No RAM test or timing change was performed. Independent immediate
copy/readback tests passed 12 GiB on both cards, and smaller allocation tests
passed on the upper card, so these results do not establish a failed VRAM chip.
The GPU-side probe then reported eight identical source/destination pairs
64/96 after the copy kernel: the GPU itself reads the correct host-USM byte
and the changed device byte. Thus the mismatch precedes CPU readback and
attention calculations. This localizes a device write/storage fault; it does
not by itself distinguish faulty hardware from driver/addressing issues.
The upper-card baseline is excluded from speed comparisons until corrected.

The lower A770 passed dense/native upload verification, a fixed 10K greedy
continuation with requested top-20 logprobs, and sampled/agent streaming cases
(two runs each). It is a temporary one-card recovery, not validation of the
original two-card throughput or a completed 128K stability run. Work 3 and the
remaining full dual/single gates are pending a correct upper-card baseline.

Recovery verification with the patch-60 build (`test-logs/recovery60-check.txt`):
fixed 10K greedy/top-20 response passes; sampled code and streaming agent pass
both runs. The sampled-code rate is 10.0–10.7 tok/s and agent 15.7–16.9 tok/s on
physical GPU1; these short sampled cases are not matched to the old dual-card
34.65 tok/s greedy profile. During this verification the lower card reaches
60 C / 1307 rpm, with no thermal flags or new kernel errors; the upper card is
idle (44 C). OpenCode CLI returns the correct color, "Красный", for a supplied
red image using the same authenticated endpoint. Its 131072 context setting is
advertised by health, but no new full-capacity stability claim is made here.

The running transient unit is `arc-moe-gpu1-recovery.service` on 8798. It selects
physical GPU1 and enables serial initial cache fill, pinned native startup
upload and byte verification; the new throughput experiments remain off.
The deliberate i915 reset at 11:50:32 is recorded as a GPU HANG in the journal;
that entry is from the controlled reset, not evidence of a spontaneous hang.
No old server on 8790 was started. Further dual-card measurements and default
promotion require resolving the reproducible upper-card upload fault first.

### Bounded upload verification, patches 61–63 (2026-10-09; unresolved)

These patches are diagnostic containment, not a validated performance fix.
`STRATA_UPLOAD_VERIFY=0` and `STRATA_UPLOAD_RELOCATE=0` remain the shared defaults.
No new performance default was promoted, and no new full profile was run.
All three patched builds compiled successfully on the server.

Patch 61 verifies dense/native uploads against the original CPU bytes using a
bounded 8-MiB host-USM readback buffer. Each failing chunk has at most two extra
copies (three attempts total); an unresolved mismatch throws before publishing
that tensor. Patch 62 optionally quarantines failed native allocations and
tries at most two replacements, with a 128-MiB quarantine limit. Only verified
replacements are published. It is not a working workaround: the engine loaded
once but still returned NaN on a 10K request. Both controls are startup costs,
not speed improvements. With verification disabled the original upload path
remains available without rebuilding.

Patch 63 applies the same verification/retry to blocking expert-cache uploads
and verifies a final snapshot of every native matrix against the original GGUF
mapping before publishing those native tensors. Serial initial cache fill
(`STRATA_FILL_SERIAL=1`) is necessary to cover every initial cache upload with
this blocking path. Queued/pipelined cache updates are not covered by this
change, and the native snapshot is taken before the later cache fill.

Targeted upper-card results, physical GPU0 / PCI 03:00.0:

| Attempt | Result |
| --- | --- |
| Disable USM/buffer compression; regular command lists | Native upload still differs: expected 57, read 25 at byte 1986329 of blk24.ssm_out.weight. |
| Isolated Compute Runtime 26.35.39758.10 (L0 reports 1.17.39758+10) | Native upload still differs: expected 72, read 104 in blk10.ssm_out.weight. No system driver package was changed. |
| Patch 61 readback + two retries | blk11.attn_q.weight byte 7639379 remains 76 -> 108 after all three attempts. |
| Patch 62 bounded allocation quarantine | Three chunks repaired; one 12,902,400-byte allocation replaced. Engine loads, but 10K request fails: last row has 10240/10240 nonfinite values. Short 7*9 request also fails semantically. |
| Full PCI function reset (14:31:39 +05:00) | Subsequent startup repairs 18 chunks, but the 10K request still fails with NaN. |
| Upper GPU core capped at 1800 MHz | Upload retries allow startup; 10K still fails with NaN. Both cards restored to 2400 MHz maximum afterward. |
| Patch 63 native snapshot + serial cache verification | All 300 native tensors pass final snapshot. Startup then fails during expert-cache profile fill at pair 6868: byte 317779 remains 0 -> 32 after three attempts. |

The final patch-63 log records 87 mismatch events and 74 repaired chunks over
native/cache initialization. Events include retries; this is not a count of 87
independently damaged allocations. The terminal error is:
`ExpertCache::fill_slot_blocking: upload readback mismatch remains after bounded retries`.
See uncommitted `test-logs/strata-verified63-gpu0.log` for exact sequence.

Two independent 12-GiB VRAM retention probes passed on the upper card: uniform
0x40/0xdf fills, and distinct pseudorandom data per chunk with all writes
completed before a delayed full readback. Earlier immediate copy checks also
passed. These do not reproduce the model's allocation/command pattern and do
not certify the hardware as healthy. Conversely, a reproducible wrong device
byte is not sufficient to diagnose a physically defective VRAM chip. The fault
could still involve device addressing/storage, driver residency or hardware;
its root cause remains unresolved. Earlier GPU-side comparison confirmed the
changed device byte while the host-USM source byte remained correct, so this
is not merely an OpenCode rendering problem or CPU readback artifact.

No RAM test/timing adjustment, BIOS change, undervolt, system driver installation,
full performance profile, or old 8790 service was performed in this continuation.
A related upstream runtime residency bug was considered, but the newer runtime
and regular command lists still reproduced this fault; it is not established
as that issue. No further optimization or tests were started after the user's
stop instruction.

Final service state: transient `arc-moe-gpu1-recovery.service` on 8798, physical
GPU1 (lower card / PCI 0b:00.0), now using the patch-63 build with verification
and quarantine controls off. Health reports loaded=true, authenticated API,
images=true and max_context=131072. This is a health check, not a new patch-63
sampled/agent or full-128K stability test. Functional recovery tests quoted
above were run on patch 60. The transient unit is not a reboot-persistent setup.
The upper card is excluded from inference; new dual-card correctness/speed
claims are blocked by its unresolved upload fault. Optimization is stopped.
