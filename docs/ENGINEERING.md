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

## What was verified here, and what was not

| Check | Status |
|---|---|
| Host + device compile of all 84 SYCL TUs at `a5682fe` + patches | done (icpx 2026.1.1) |
| FP64 scan of the device code, `STRATA_SYCL_NO_FP64=ON` | 0 instructions |
| Every per-kernel SPIR-V module through `ocloc -device acm-g10` | 994 of 1001 before patch 6; the 7 failures are exactly the `joint_matrix` modules |
| Host link of the engine | fails at `a5682fe` (#871 drift); links with patch 5 |
| AOT build of `strata` for `acm-g10` with all 6 patches | **links**: 994 device modules built by IGC for acm-g10, 0 compiler errors (the 7 XMX modules are stubs now), 16 min on 4 cores; the 102 MB binary runs (`--help`) |
| A tiny double kernel: `acm-g10` fails, `bmg-g21` builds | reproduced |
| Run on an A770 | **not yet**: no Arc in the build machine. `./arc bench` reports the real numbers |
| Kernel parity tests (`ARC_PARITY=ON`, `ctest`) on an A770 | not yet. Expect the tight-tolerance ones that compare against float64 references to report ulp differences |
