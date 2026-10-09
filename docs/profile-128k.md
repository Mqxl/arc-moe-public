# 128K + vision baseline and profile — 2026-10-09

Parent 8115b48; 63 engine patches. This is stage 0: no engine or performance-default change. The parent gitlink remains `a5682fea1667c87b9a6b9ad0b2cd69dd9898fbaf`.

## Configuration and method

Configuration: `strata-tuned-opencode.json`, 131072 maximum context, int8 KV, 32768 resident KV, CPU vision encoder (4 threads, 512 image-token cap), existing IQ2_XS shards and multilingual MTP union. `STRATA_UPLOAD_VERIFY=1`, serial/pinned verified upload remain enabled. Two A770 16 GiB: upper `03:00.0`, lower `0b:00.0`. No RAM tests or settings changes.

Dual uses `tune-dual.env`: spec 6, short-read 64, borrowed prefill with 4096-token chunks, reserve 768 MiB. Single uses `ARC_GPUS=0` and the empty `tune-single.env`, retaining auto prefill (8192-token chunks). Both use the same 128K + vision model configuration. The historical 32K tuning run is not this baseline.

Command: `python3 tools/run-authorized.py third_party/strata/strata-tuned-opencode.json tools/bench.py --url http://127.0.0.1:8798 --cases short,prose,10k,32k,64k,120k,agent,vision --runs 2 --max-tokens 128 --ttft --vram --show --json <output>`; run with the existing secret EnvironmentFile, without a key in OS arguments.

New mixed fixtures have independent deterministic records and a unique request seed. Input length is counted through the serving tokenizer; completion usage is authoritative (template differences explain the small offset). Actual mixed input lengths: about 31945–31948, 63953–63960, 119952–119955 tokens. Fresh long requests reused zero cached tokens. TTFT starts before the HTTP call and stops on the first productive SSE payload, excluding empty role chunks. Engine prompt/decode timing and client TTFT/wall time are distinct.

Vision uses a dominant-red PNG with seed-specific corner pixels and 2K mixed text. Every run has a different image hash, avoiding the server image-embedding cache. Original identical-image rows were preserved in `stage0-{dual,single}-original.json`; only vision rows in the final baseline JSON were replaced by the separately saved `stage0-*-vision-fresh.json` runs.

VRAM is the engine clients’ resident local DRM memory, sampled every 100 ms, deduplicated by PCI BDF/client id. These are observed peaks in MiB, not exact instantaneous maxima or Sysman free-memory estimates. Maximum context is configured at 128K; the largest exercised fresh input is 120K, not a claim that every possible 128K workload fits.

## Untraced baseline

Values are medians of two runs; peak VRAM is the maximum of those runs. All 16 requests in each mode passed nonempty/repetition/error checks; both modes passed the complete 14-request correctness check.

| Mode | Case | Prompt tok/s | TTFT s | Decode tok/s | Wall s | Upper / lower peak MiB |
|---|---|---:|---:|---:|---:|---|
| dual | short code | 49.55 | 0.730 | 33.20 | 4.530 | 14942.1 / 15048.2 |
| dual | short prose (ru) | 52.50 | 0.808 | 27.10 | 5.478 | 14942.3 / 15048.4 |
| dual | 10K code | 506.75 | 21.193 | 28.80 | 25.593 | 15027.3 / 15131.4 |
| dual | 32K mixed code | 549.20 | 58.309 | 28.10 | 62.813 | 15027.6 / 15131.6 |
| dual | 64K mixed code | 556.45 | 115.154 | 27.10 | 119.825 | 15027.6 / 15142.3 |
| dual | 120K mixed code | 545.40 | 220.294 | 27.95 | 224.823 | 15028.7 / 15134.8 |
| dual | agent (stream) | 103.50 | 3.424 | 36.75 | 4.756 | 14945.5 / 15052.6 |
| dual | vision + 2K | 315.25 | 6.461 | 31.85 | 10.426 | 15010.2 / 15096.2 |
| single | short code | 36.35 | 0.982 | 24.30 | 6.195 | 15002.1 / inactive |
| single | short prose (ru) | 35.90 | 1.165 | 19.85 | 7.552 | 15013.3 / inactive |
| single | 10K code | 417.70 | 25.700 | 20.65 | 31.843 | 15149.2 / inactive |
| single | 32K mixed code | 436.35 | 73.349 | 20.35 | 79.596 | 15151.2 / inactive |
| single | 64K mixed code | 429.65 | 149.089 | 19.35 | 155.651 | 15152.5 / inactive |
| single | 120K mixed code | 415.05 | 289.390 | 19.80 | 295.787 | 15183.4 / inactive |
| single | agent (stream) | 69.50 | 5.079 | 27.00 | 6.629 | 15017.4 / inactive |
| single | vision + 2K | 304.85 | 6.704 | 23.40 | 12.118 | 15079.0 / inactive |

## GPU timeline: dual, same configuration

`unitrace --chrome-kernel-logging --chrome-call-logging --verbose -d --teardown-on-signal 10` collects from process start. A paused run produced no replay events and was discarded; continuous collection produced 564476 GPU events. Request markers select a complete 48-layer verify body after the first productive token. Sampler boundaries prevent accidentally joining adjacent windows or confusing MTP samplers with verifier windows. The prefill interval is one steady 4096-token owner chunk near the end of 64K, with concurrent neighboring peer-chunk activity visible.

Times are interval unions, not sums of overlapping operations. Host waits overlap GPU execution: they are not additional critical-path cost. Idle means absence of recorded operations; kernels spinning on flags are classified separately and count as occupied. Trace overhead means these intervals do not replace the untraced throughput table.

| Interval | Span ms | Any-GPU union ms | No recorded GPU op ms | Host sync union ms (overlaps GPU) |
|---|---:|---:|---:|---:|
| short verify / verify_window_after_first_token | 137.632 | 126.265 | 11.367 | 135.534 |
| 64K prefill and verify / verify_window_after_first_token | 125.012 | 113.776 | 11.236 | 122.918 |
| 64K prefill and verify / prefill_pipeline_interval | 7368.610 | 7189.475 | 179.135 | 1840.304 |

| Interval / GPU | Attention % | MoE % | DMA H2D % | PCIe fetch kernel % | Handoff DMA % | All mapped-copy kernels % | Device flag wait % | Unrecorded idle % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short verify / verify / lower | 0.23 | 4.36 | 0.00 | 0.02 | 0.00 | 0.11 | 0.06 | 83.01 |
| short verify / verify / upper | 0.46 | 48.00 | 0.00 | 0.04 | 0.00 | 0.07 | 0.11 | 25.25 |
| 64K prefill and verify / verify / lower | 0.77 | 5.06 | 0.00 | 0.02 | 0.00 | 0.13 | 0.07 | 77.31 |
| 64K prefill and verify / verify / upper | 1.38 | 34.77 | 0.00 | 0.05 | 0.00 | 0.08 | 0.12 | 31.68 |
| 64K prefill and verify / prefill / lower | 6.15 | 21.16 | 5.00 | 0.00 | 0.44 | 0.00 | 0.00 | 57.75 |
| 64K prefill and verify / prefill / upper | 12.32 | 47.70 | 13.90 | 0.00 | 0.17 | 0.00 | 0.00 | 4.56 |

Prefill handoff is a 160 MiB residual (`T=4096`, `D=4×2560`, fp32), copied through pinned host memory. The matching-size DMA attribution is based on `prefill.cpp`, not a guess that all copies are P2P. Decode handoff uses the `copy_from_mapped` kernels over mapped pinned host memory (`verify.cpp`). Those kernels also serve PLE/history/expert output, so the table reports their total as an upper bound on decode handoff, not an invented exact split. No direct GPU-to-GPU P2P transfer appears in this path. MoE kernel duration can include mapped weight reads; the table does not claim all PCIe traffic occurs in DMA/fetch rows.

## Native recording capability

`tools/probe-native-graph.cpp` isolates graph construction and one trivial device kernel, without weights, shared allocation, oneMKL or host operations inside capture. With `SYCL_GRAPH_FORCE_NATIVE_RECORDING=0`, both devices produce 63. With 1, both fail at **construction**, before `begin_recording` or any kernel: `sycl` error code 8, `Device does not support graph record and replay feature (UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP)`. Explicit Level Zero V1 and V2 produce the same result. This is a capability limitation of the installed runtime/backend, not evidence of an illegal operation in the verify body or an inability to execute ordinary compiled SYCL graphs. Keep native recording off; do not claim fallback to ordinary graphs as native acceleration.

Intel documents that [SYCL_GRAPH_FORCE_NATIVE_RECORDING](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md) forces every command graph into the optional native recording mode. The local probe, not the newer online documentation, establishes this installed runtime’s behavior.

## What to optimize next

1. The 46 ms placement prediction is not measured GPU kernel time. The observed short verify body is 137.6 ms, with 126.3 ms of recorded GPU operations and only 11.4 ms without a recorded operation. Host sync union of 135.5 ms mostly watches the same GPU work; removing a supposed additional 100 ms of host waits is not supported by this trace.
2. Upper-card MoE dominates: about 66.1 ms of the short window. Its top gate/up and down kernels account for 30.4 and 20.3 ms, plus 13.3 ms for another gate/up format. The peer’s corresponding kernels cost about 2.9/2.7 ms. Investigate residency, format/kernel work and the 34/14 layer distribution before speculative new handoff spins.
3. The owner prefill interval is 95.4% occupied; MoE accounts for 47.7%, attention 12.3%, H2D DMA 13.9%. Lower-card idle is 57.8%, so a measured layer balance or expert-sharing improvement may have room. Existing chunk pipelining is already active. This trace does not prove PCIe x4 is a physical throughput ceiling or that prefetch/XMX would win.
4. Int8 KV and split-K decode attention already exist in this baseline. Verify attention contributes under 2% on the owner after 64K, whereas prefill attention is more substantial. Do not present duplicate implementations as new gains.
5. Vision encoder timing needs a separate measurement: TTFT alone does not isolate encoding. Stage 0 establishes fresh-image latency, not an encoder optimization.

## Saved evidence

Raw files remain uncommitted in `test-logs/`: `stage0-dual.json/.txt`, `stage0-single.json/.txt`, both `stage0-*-check.txt`, original and fresh vision rows, `stage0-trace-requests.json`, `stage0-trace-summary.json`, `stage0-trace-active/strata.39414.json`, and `graph-capability-*.txt`. Secret values are not included. Baseline model/config files remain host-specific and are not added to Git.

Config SHA256: `c8c5cf9716556dcf886a1833a904014ebfa975f88e32e681dded1ffbdd4d9d6e`; dual tune SHA256: `a63dfbafdb89edba60f894fdfe06d2b4f19d1d2fd2788510795a2f33dcc11ad5`; single tune SHA256: `5a708b5ad3c3a2529bc124053568ab56e9967d3bc9f23793a3b23a944d66d9c3`.
