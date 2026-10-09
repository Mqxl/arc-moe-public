# Requested uncensored IQ2_XS migration on two A770s

This is a requested model-weight change, not an accepted engine speedup or a
new shared default. The stock-model optimization and dual/single controls are
in [final-128k.md](final-128k.md). The original model and pack remain intact.

Source: [SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
revision eec4e10a2c29b440a2ae85591fa11569ce752963, IQ2_XS. The publisher
transplants residual-output projections into the GSQ-RCO base. This report
validates load, transport and the cases below; it does not independently
certify the publisher's refusal-removal or broad capability claims.

Downloaded shard 1: 39,665,015,072 bytes, SHA256
4acc3ebb5050d20117bdc77fc39200e709dfe2b2367f3522c95b9112a30aff81.
Shard 2: 28,800,138,432 bytes, SHA256
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113.
Both verified. The existing matching shard 2 is reused by symlink. Total GGUF
bytes 68,465,153,504; only shard 1 was downloaded. The existing BF16 vision
projector and stock multilingual MTP runtime were reused. A separate native
pack was built; the old pack's expert offsets were not reused.

Local configuration: strata-uncensored-opencode.json, port 8798,
max_context 131072, int8 KV, 32768 resident KV, layer boundary 30/18,
prefill chunk 4096, spec 6, short-read 64, upload verification ON. No hardware
settings, engine kernels or shared defaults changed for this migration.
Native upload final snapshots passed for all 187 + 113 tensors on the cards.

## Measurements

Five cases, two runs each, max output 128, same corpus seed as the final stock
control. Throughput, TTFT and wall are medians; resident VRAM is the maximum
sampled per GPU across the two runs, upper/lower in MiB. This is a small speed
comparison, not a full new tuner or prolonged stability run.

| case | stock prefill → uncensored tok/s | stock decode → uncensored tok/s | uncensored TTFT s | uncensored wall s | VRAM MiB |
|---|---|---|---|---|---|
| short code | 56.15 → 55.05 | 37.45 → 34.45 | 0.660 | 4.325 | 14932/15051 |
| short prose (ru) | 55.65 → 54.90 | 27.60 → 25.45 | 0.775 | 5.749 | 14935/15059 |
| 10K code | 567.65 → 565.00 | 31.45 → 29.45 | 19.016 | 23.313 | 15019/15143 |
| agent (stream) | 153.15 → 152.20 | 43.35 → 35.25 | 2.344 | 3.492 | 14936/15059 |
| vision + 2K | 336.15 → 336.60 | 35.00 → 35.05 | 6.090 | 9.697 | 14998/15067 |

One fresh long-context smoke: 119951 input tokens, 128
output tokens. Prefill 628.2 tok/s; TTFT 191.307 s;
decode 33.6 tok/s; wall 195.063 s. Peak resident VRAM
{'0000:03:00.0': 15021.31, '0000:0b:00.0': 15145.65} MiB. A single observation is not a stable long-context
throughput estimate. It verifies a ~120K prompt within the 131072 configuration.

Short code is ~8% slower than stock; 10K prefill is nearly unchanged. Agent
output is short and differs across models, so its decode rate is especially
variable. The dense projections differ and resident expert slots are 8922
versus stock 9116; no new trace was collected to attribute the slowdown.
These weights fail the stock-engine 2% no-regression speed gate, and are used
locally because the user explicitly requested the uncensored model.

## Correctness, OpenCode and current service

Full check 14/14, including both 10K prompts. All 11 benchmark replies valid,
including the red image fixture; no broken/repeated/empty reply. OpenCode
1.18.35 in <client-project> returned 63 and Красный with the new model ID.
CLI verification used --pure, external MCP/plugins disabled, tools denied.
The actual project settings were otherwise retained, including its existing
other providers and default model. Vision is image input; no audio/video or
large-photo accuracy is certified here.

Project Qwen model ID: arc-moe/qwen3.8-flash-next-uncensored-iq2_xs.
Context 131072, output 4096, compaction/pruning enabled with 8192 reserved
and recent tokens. API key remains an environment reference; no key in files
or OS argv. Backup: <client-project>/opencode.before-uncensored-20261009.json.bak.
The generated standalone client config is opencode-uncensored-128k.json in
local outputs/final-128k. Start a new chat and select the new Qwen model.

Service arc-moe-uncensored is active on 8798; 8790 is not listening. The stock
model can be restored by stopping that service and running ./arc start
tuned-opencode with the same protected EnvironmentFile authorization. The
uncensored launcher is ./arc start uncensored-opencode. Configs/packs are
local generated artifacts and not committed as machine-specific defaults.

The initial measurement wrapper failed before any bench request because its
runpy path omitted tools/; the wrapper was corrected and only speed/long
measurements rerun. This was not a model or engine failure. Raw error retained
as test-logs/uncensored-speed-wrapper-error.txt.

Observed max temperature during checks/measurements: upper 86 C,
lower 76 C; 2 s sampling, no fan/voltage/power/clock changes. No GPU
hang/reset or OOM in the kernel log during this migration.

At the user's request only /srv/models/minimax-m2.7 and /srv/models/mimo
were deleted, freeing ~237 GiB. After the new shard, /srv/models has ~283 GiB
free, root ~216 GiB free; /srv/storage remains almost empty (~445 GiB free).

Raw reports, not committed: check-uncensored-20261009.txt,
test-logs/uncensored-speed.{json,txt}, uncensored-120k.{json,txt},
uncensored-temperature.jsonl, download-uncensored.log, pack-uncensored.log.
