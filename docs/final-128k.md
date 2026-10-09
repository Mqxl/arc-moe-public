# 128K + vision optimization report

Optimization was stopped under the user's criterion after the remaining dual
candidates failed the speed gate. This is the fastest verified configuration
in this run, not a proof of the hardware's physical limit.

## Configuration and controls

Actual model: strata-tuned-opencode.json, max_context=131072, int8 KV with
32768 resident tokens, CPU vision (4 threads, 512 image-token cap), IQ2_XS,
MTP multilingual union, max benchmark output 128. Dual prefill chunks 4096;
single auto chunks. Upload verification remains enabled on every engine load.
No RAM test, BIOS, clocks, voltage, power or fan setting was changed.
8790 was not started. The API key is supplied from EnvironmentFile and is not
present in these reports, config artifacts or operating-system argv.

Stage-0 baseline and 32K/64K/120K inputs are in docs/profile-128k.md.
The largest exercised text input is about 119952 tokens; the configured ceiling
is 131072. This does not certify every possible full-limit tools/image session.
The existing OpenCode project declares 131072 context, 4096 output, automatic
compaction/pruning with 8192 reserved tokens and an environment key reference.
Server fit_max_tokens is enabled. No unrelated OpenCode settings were edited.

## Confirmed step

43255ff: optional layer boundary override, local 30/18 versus auto 34/14.
Dual full check 14/14; single full check 14/14. All eight dual speed/latency
cases passed the 2% comparison gate. Single override is ignored and launcher
argv is byte-identical; five-case control remained within 2%.
Dual 10K prefill 506.75 ->568.25, 64K 556.45 ->644.9, 120K 545.4 ->630.35;
short code decode 33.2 ->37.6 tok/s. Machine boundary is not a shared default.

## Profile conclusions and limits

Stage-0 complete short/64K verification windows: 137.632/125.012 ms.
GPU busy unions: 126.265/113.776 ms, uncovered gaps: 11.367/11.236 ms.
Host waits overlap these intervals; adding waits to GPU time is invalid.
MoE is the largest part and includes reads of mapped host expert weights.
Baseline upper/lower idle: short 25.25%/83.01%, 64K 31.68%/77.31%.
This justified measured layer balance. The old predicted 46 ms is a placement
estimate, not a measured sum of current verification kernels.

Native Level Zero recording fails at graph construction before any body is
recorded: UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP unsupported,
SYCL error 8. Minimal write-63 graph replay works with native forcing off on
both GPUs and fails when forced with both installed adapters (V1 and V2).
This is a runtime capability restriction; it is not proof of an illegal host
operation inside recording, nor a physical ceiling of A770. Ordinary SYCL
verification graphs remain enabled. No native-recording speedup is claimed.

Split-K decode attention and int8 KV already exist in this build. Attention
accounts for about 1–2% of the measured verify window; prefill attention about
12% of the owner chunk. Existing FP16/XMX experiments and early prefetch losses
are documented in ENGINEERING; no previously rejected switch was made default.
Handoff currently uses pinned host memory, not P2P. Prefill transfer is about
160 MiB per 4K chunk; decode mapped-copy time in the trace is a small upper
bound that also includes unrelated mapped copies. No PCIe hardware ceiling
has been inferred from aggregate MoE time.

## Vision

Two fresh uncached benchmark images: encoder protocol reports 124/123 ms,
wall 123.983/122.856 ms (9 image tokens each). At dual TTFT 6.096 s this is
about 2%; moving this encoder to a GPU is not justified by this fixture.
The small benchmark image does not establish encoder costs for large photos.
Vision encoder/projector models were reused, not downloaded again.

## Rejected experiments

Wide expert tails (8-entry tile) match the tested numerical reference but
upper GPU VRAM timing worsens: ~0.380 ->0.410 ms for ten experts/six entries;
mapped-host weights ~1.619 ->1.599 ms. It stays opt-in, mode 2, not selected.
The narrower three-entry tail (mode 1) passed exact real-weight parity on both
GPUs, both storage modes, and 5/6/7-entry synthetic groups after correcting a
reference-index mistake in the test itself. An early different-seed
code improvement did not repeat: final same-seed 37.45 ->37.60 (+0.40%),
10K 567.65 ->564.7. No target gain >=2%, so mode 1 also stays OFF.

Global spec5 screen: code 37.6 ->36.2, post-10K 32.0 ->32.6, post-64K
30.45 ->30.5; agent decode 40.65 ->29.5. No meaningful long-context improvement
was demonstrated here, so a context-adaptive spec5 policy was not introduced.

Final tune shortlist: expert_tail, commit_overlap, split_rows8. Base/final
matrices include all eight cases; screens include short, prose,10K,64K,agent,
vision. Every metric is compared; agent screen tolerance 3%, final 2%. JSON
must contain complete distinct rows with valid nonempty replies and finite
positive timings/rates. Fallback is rechecked against the base. Each mode
resolves actual config/tune values independently, including config precedence;
an opt-in STRATA_CONFIG_ENV_OVERRIDE lets a candidate reach the engine.

The dual shortlist completed: commit overlap short decode +0.93%, agent wall
+3.50%; split_rows8 10K prefill +1.26%, post-10K decode -3.82%, agent decode
-6.69%; expert tail +0.40%. All were rejected and the verified baseline saved.
Single base completed full 14/14 plus all eight cases twice. Further single
candidates were intentionally cancelled only after the engine had stopped,
per the user's instruction; this is not a completed full single tune sweep.
No kernel default or machine-specific layer boundary was promoted globally.

## Final controls and measured results

Medians of two requests per case. Resident VRAM is the maximum observed
per-card peak across those two requests (DRM fdinfo, 100 ms sampling), not
total host memory or a guarantee of the exact instantaneous maximum. Dual
VRAM order is upper 03:00.0 / lower 0b:00.0; single is the upper card. Unique
seeds create slightly different mixed inputs; comparison is not bit-identical
input parity. All 16 rows in both final controls were valid.

### dual: stage 0 → final control

| case | prefill tok/s | TTFT s | decode tok/s | resident VRAM peak MiB |
|---|---|---|---|---|
| short code | 49.55 → 56.15 | 0.73 → 0.65 | 33.20 → 37.45 | 14942/15048 → 14934/15054 |
| short prose (ru) | 52.50 → 55.65 | 0.81 → 0.76 | 27.10 → 27.60 | 14942/15048 → 14938/15062 |
| 10K code | 506.75 → 567.65 | 21.19 → 18.93 | 28.80 → 31.45 | 15027/15131 → 15021/15156 |
| 32K mixed code | 549.20 → 626.95 | 58.31 → 51.10 | 28.10 → 31.80 | 15028/15132 → 15024/15156 |
| 64K mixed code | 556.45 → 644.20 | 115.15 → 99.48 | 27.10 → 31.40 | 15028/15142 → 15024/15157 |
| 120K mixed code | 545.40 → 629.75 | 220.29 → 190.83 | 27.95 → 31.55 | 15029/15135 → 15024/15159 |
| agent (stream) | 103.50 → 153.15 | 3.42 → 2.33 | 36.75 → 43.35 | 14945/15053 → 14941/15068 |
| vision + 2K | 315.25 → 336.15 | 6.46 → 6.10 | 31.85 → 35.00 | 15010/15096 → 15013/15076 |

Agent wall: 4.756 → 3.275 s.

### single: stage 0 → final control

| case | prefill tok/s | TTFT s | decode tok/s | resident VRAM peak MiB |
|---|---|---|---|---|
| short code | 36.35 → 36.30 | 0.98 → 0.98 | 24.30 → 24.60 | 15002 → 15005 |
| short prose (ru) | 35.90 → 35.85 | 1.16 → 1.17 | 19.85 → 20.05 | 15013 → 15013 |
| 10K code | 417.70 → 416.65 | 25.70 → 25.76 | 20.65 → 20.95 | 15149 → 15149 |
| 32K mixed code | 436.35 → 435.75 | 73.35 → 73.46 | 20.35 → 20.35 | 15151 → 15182 |
| 64K mixed code | 429.65 → 428.35 | 149.09 → 149.52 | 19.35 → 19.40 | 15152 → 15163 |
| 120K mixed code | 415.05 → 414.25 | 289.39 → 289.94 | 19.80 → 20.70 | 15183 → 15153 |
| agent (stream) | 69.50 → 69.35 | 5.08 → 5.09 | 27.00 → 28.45 | 15017 → 15017 |
| vision + 2K | 304.85 → 303.15 | 6.70 → 6.75 | 23.40 → 23.55 | 15079 → 15089 |

Agent wall: 6.629 → 6.639 s.


## Verification, thermals and reproducibility

Full correctness: dual and single 14/14, including both 10K requests, sampled
and streamed agent cases. Text benchmarks exercised up to ~119953 input
tokens with max output 128, using the actual 131072-configured int8-KV server.
The existing project <client-project> OpenCode 1.18.35 client returned 63
for 7*9 with no transport errors. This used --pure with external MCP/plugins
disabled and tools denied; it does not certify every project plugin or UI flow.
OpenCode image attachment also correctly identified the fresh red fixture.
Vision benchmark identifies the red fixture; large photos, audio and video
were not validated by this final matrix.

Observed maxima during the final tune/control: upper 87 C, lower 77 C,
upper fan 899–3505 rpm and lower 1019–1498 rpm. Sensor samples every 2 s;
the initial short observer segment omitted upper readings, so these are
observed maxima. No fan, voltage, clock or power limit was changed. Sustained
stability beyond these completed runs is not claimed.

Commits: 43255ff accepted optional layer balance; 06160c6 rejected opt-in
expert-tail experiment plus parity harness correction; 8c6b263 effective
config precedence, per-GPU baseline isolation and stricter tuner acceptance.

Reports are intentionally uncommitted: tune-20261009-190836.txt,
.tune-20261009-190836/{dual,single}-base.json and *.check.txt,
check-final128k-{dual,single}.txt, test-logs/final-temperature.jsonl,
test-logs/stage0-trace-dma-summary.{json,txt}. The root profile reports and
test-logs/stage0-trace-active/strata.39414.json supply baseline trace evidence.

Service arc-moe-tuned-balanced on 8798 restored the verified stock IQ2_XS
configuration with local 30/18 and 4K chunks, max_context 131072, images enabled
and API authentication. A separate requested uncensored-model migration is
not included in the numerical tables above. Its weights and measurements
must be reported separately; it cannot inherit stock-model acceptance.


The re-parsed baseline prefill interval contains 6138 owner H2D expert copies
of 1,510,400 bytes at aggregate 14.22 GB/s and 1020 peer copies of the same
size at 4.99 GB/s. Other 1,305,600-byte owner copies aggregate 14.31 GB/s;
1,177,600-byte peer copies 5.00 GB/s. These rates use complete event bytes
and union durations, separately per GPU/size/direction; they are observed
transfers, not a claim that end-to-end MoE compute reaches those rates or
that software can turn the lower x4 slot into a 25 GB/s link. The 160 MiB
owner D2H handoff event takes 12.531 ms (~13.39 GB/s), using pinned host RAM.

Current requested model migration and active service: [uncensored-iq2-xs.md](uncensored-iq2-xs.md). Stock before/after tables above remain stock-model measurements.
