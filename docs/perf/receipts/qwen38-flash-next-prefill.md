# Qwen3.8 Flash-Next long-context prefill receipt (qwen4_exp)

This is the scrubbed, tracked receipt for the measured long-context prefill behaviour of the two
Qwen3.8 Flash-Next packs. The raw probe artifacts remain local (`~/.mtplx/bench/`, also mirrored to
the excluded `agent-output/long-context-prefill/`); their hashes are recorded at the bottom so the
result can be audited without committing bulky per-request logs.

Plan of record for the fixes: `docs/plans/2026-09-06-long-context-prefill-handoff.md`.

## Code and machine provenance

- Measured against the **installed** runtime: `mtplx 2.11.0` (`~/.mtplx/bin/mtplx --version`),
  MLX 0.32.2, Python 3.14.5, macOS 26.4.1.
- Benchmarked tree at measurement time: this checkout at `d0518c9` on `issue-308`, version
  **2.9.0** — which did **not** contain the `qwen4_exp` descriptor, the guard at
  `mtplx/generation.py:1205`, or the KV-geometry constant. The numbers below therefore describe
  the installed 2.11.0, not that tree.
- After the 2026-09-06 rebase the tree is `main` = `406b5f768e984e036d16aca1edaddaa29fe8519e`
  (2.11.1) with the DFlash2 commit replayed as
  `d6e4a5a4a63b6a3f7b4bb179d09fc0c13585a67c`. Re-baseline before judging any fix.
- Machine: Apple M3 Ultra, 32 CPU (24P/8E), 80-core GPU, 512 GB unified
  (`hw.memsize` = 549755813888). `mtplx hardware` warns this is a high-bandwidth GPU path, not an
  M5 TensorOps path. Fan policy: Apple default (`--fan-mode default`); no ThermalForge max-fan run
  was used, so per `CONTRIBUTING.md` none of these numbers is a product headline claim.
- A second, unrelated Metal server (`ds4-server`, GLM-5.3-Flash-Q4_K on 127.0.0.1:9000,
  161.4 GiB RSS) was resident throughout. It is not part of the measurement and is not excluded
  from it — a caveat that matters for the arm-to-arm comparisons below.

## Method

`~/.mtplx/scripts/prefill-probe.py` drives the **serve path** over HTTP: for each context rung it
sends one salted prompt (cold) and then one ~600-token turn appended to the same prefix
(follow-up), with `max_tokens=1` so the measurement is prefill plus first token. Results come from
the engine's own per-request JSONL (`prompt_eval_time_s`, `new_prefill_tokens`, `cached_tokens`,
`peak_memory_bytes`), read from a recorded byte offset — not from client-side timing.

Row `kind` is derived from the engine's `cached_tokens`, never from the probe's intent. An earlier
revision of the instrument mislabelled growth rungs as cold and understated the 103 k rate by 2×
(it reported 191 tok/s where the artifact says 383); that correction is why this section exists.

**Second correction, recorded because it is the more instructive mistake:** the first version of
this receipt's 9002 table transcribed the probe's *client-side wall times* (printed to stdout as
`wall=…s`, which include HTTP, tokenisation, SSE teardown) instead of the engine's
`prompt_eval_time_s`. The two differ by up to 18 % (the 103 k follow-up is 2.397 s in the engine
and 2.83 s at the client), which silently inflated the headline decay from 2.50× to 2.9× and moved
the derived KV geometry from 249,526 to 244,249 B/token. Every number here now comes from the
JSONL engine rows, and the tables state `prompt_eval_s`, not wall.

`--tag` seeds the prompt salt, so a new tag is a genuinely cold prompt set. Throughput below is
`new_prefill_tokens / prompt_eval_time_s`.

## Baseline 9002 — `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`

Profile `turbo`, `--depth 3`, `MTPLX_NGRAM_RESIDENT=1`, `--context-window 262144`,
`--prefill-chunk-tokens 2048`, `--scheduler-mode serial`, `--paged-kv-quantization off`.

| kind | prompt | new | cached | prompt_eval_s | new tok/s | peak |
|---|---|---|---|---|---|---|
| cold | 6,500 | 6,500 | 0 | 8.012 | 811 | 112.6 GiB |
| prefix_hit | 7,049 | 556 | 6,493 | 0.960 | 579 | 112.6 GiB |
| cold | 25,745 | 25,745 | 0 | 27.965 | 921 | 117.7 GiB |
| prefix_hit | 26,294 | 556 | 25,738 | 1.220 | 456 | 117.7 GiB |
| cold | 51,405 | 51,405 | 0 | 69.276 | 742 | 124.1 GiB |
| prefix_hit | 51,954 | 556 | 51,398 | 1.615 | 344 | 124.1 GiB |
| cold | 102,731 | 102,731 | 0 | **267.920** | **383** | 135.0 GiB |
| prefix_hit | 103,280 | 556 | 102,724 | **2.397** | **232** | 135.0 GiB |

## Baseline 9001 — `grant-ai/Qwen3.8-Flash-Next-Abliterated-MTPLX-4bit`

Profile `sustained`, `--depth 1`, `MTPLX_NGRAM_RESIDENT=0` (sidecar streamed), same window/chunk.

| kind | prompt | new | cached | prompt_eval_s | new tok/s | peak |
|---|---|---|---|---|---|---|
| cold | 7,478 | 7,478 | 0 | 7.579 | 987 | 85.4 GiB |
| prefix_hit | 8,150 | 0 | 8,150 | 0.030 | 0 | 85.4 GiB |
| cold | 26,727 | 26,727 | 0 | 29.358 | 910 | 90.3 GiB |
| prefix_hit | 27,399 | 679 | 26,720 | 1.496 | 454 | 90.3 GiB |
| cold | 52,389 | 52,389 | 0 | 71.039 | 737 | 95.7 GiB |
| prefix_hit | 53,055 | 673 | 52,382 | 1.897 | 355 | 95.7 GiB |
| cold | 103,714 | 103,714 | 0 | **268.637** | **386** | 104.5 GiB |
| prefix_hit | 104,382 | 675 | 103,707 | **2.835** | **238** | 104.5 GiB |

The `8,150` follow-up row reports `new_prefill_tokens: 0` — a 672-token turn that the engine
charged as a pure prefix hit at 0.030 s. It is left in the table because it is in the artifact and
because it shows the cache path is capable of returning in tens of milliseconds; the other
follow-up rows are the ones that carry cost.

## Arms

| arm | single variable | cold eval_s | cold tok/s | follow-up eval_s | peak |
|---|---|---|---|---|---|
| baseline | — | 267.920 | 383 | 2.397 | 135.0 GiB |
| A | `--context-window 131072` | 268.162 | 383 | 2.853 | 123.3 GiB |
| C | `--prefill-chunk-tokens 8192` | **315.868** | **325** | 2.757 | **157.7 GiB** |
| D | `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=32768` | 267.003 | 385 | 2.860 | 123.3 GiB |
| B | `--paged-kv-quantization q8` | refused at load, downgraded to `off` — no artifact | | | |

Arm D is a real null result, not a dead setting: the
`dense-decode ceiling auto: … tokens` line is present in the baseline serve log and **absent** when
the ceiling is supplied explicitly.

### Repetition noise — read before comparing any two follow-up numbers

The four runs above all measured the same nominal work: one ~556–679-token follow-up at ~103 k
context. Their `prompt_eval_time_s` values are 2.397 (baseline), 2.853 (A), 2.757 (C), 2.860 (D):
**median 2.805 s, stdev 0.218 s, spread 17.0 %.** The baseline's 2.397 s is the low outlier, not the
other three being slow.

Cold prefill at the same context, excluding arm C which was deliberately a different setting:
267.920 / 268.162 / 267.003 s → **stdev 0.611 s, spread 0.43 %.**

Consequence, and it changes how the defect list should be attacked:

- Cold-prefill comparisons are conclusive at the ±0.5 % level. Everything claimed about
  profile / window / chunk / ceiling on the cold path stands.
- **Follow-up comparisons are not** resolvable below ~20 % from one run each. "Follow-up unchanged"
  is justified for A/C/D against each other (2.757–2.860 s) and *not* against the baseline's
  2.397 s. The honest range for the follow-up decay versus the 0.960 s at 6.5 k is
  **2.50×–2.98×**, not a single 2.5×.
- Any acceptance test on the follow-up path needs **≥3 repetitions**, judged on the median. A
  claimed 2× win (1.40 s from the median) clears the measured stdev comfortably; a claimed 10 % win
  would be indistinguishable from this noise.

## Counters and cross-checks

- `paged_gqa_sdpa_calls`, `prefill_partitioned_paged_calls`, `prefill_dense_fallback_calls`,
  `attention_dense_fallback_calls`: **0 at every rung on both packs**.
  `prefill_attention_impl` and `prefill_layout` are absent from the rows entirely.
- On the slow long-context rows of the historical 27B log, `target_forward_time_s` is 99.7 % of
  `prompt_eval_time_s` (32.47 of 32.57 s), with `state_rebase_count: 0`,
  `trunk_cache_materialize_time_s: 0.00`, `cache_restore_time_s: 0.00` — the cost is the forward
  pass, not restore or paging bookkeeping.
- Marginal KV from `peak_memory_bytes`: 9002 **249,526 B/token** ((135.0−112.6) GiB /
  (102,731−6,500) tokens, = 244 KiB, **3.81×** the assumption); 9001 **213,309 B/token** (208 KiB,
  3.25×). The engine's dense-decode policy assumes **65,536 B/token** (`generation.py:1499`). The
  engine's own KV-quantization refusal message states "KV on 12 of 48 layers (~24 KB/token)"
  ≈ 288 KB/token, which agrees with the measurement and contradicts the assumption.
- Projection at 262,144: **172.0 GiB** (9002) / **136.0 GiB** (9001) against the 192.0 GiB engine
  budget — reachable. An earlier draft of this receipt claimed 262 k was over budget ("278.2 GiB")
  and a later one said 173.5 / 169.4; both were wrong, the first by double-counting the 107.1 GiB of
  weights already inside `peak_memory_bytes`, the second by carrying the client-wall-time figures
  described below.
- Persona tax: the identical one-line prompt is **62 prompt_tokens on 9002 and 1,058 on 9001**
  (+996); probe-body deltas were +982 and +984 at two rungs. Cause: 9001's `chat_template.jinja`
  is the Blackfrost build while its `tokenizer_config.json` is stock Qwen.
- `vm.swapusage` read `used = 0.00M` at session start and through every single-server arm; with
  both Flash-Next ports plus `ds4-server` resident it read `used = 81.56M`.

## Raw artifact hashes

First 16 hex chars of SHA-256, paths relative to `~/.mtplx/`:

```
aa95fd34d7163110  bench/prefill-probe-9001-baseline-20260905-222454.json
f015896cfc2b30e3  bench/prefill-probe-9001-gate-20260905-222412.json
055085e7d25bf7f0  bench/prefill-probe-9002-armA-ctx131k-20260905-223642.json
8433d4d265fd3f54  bench/prefill-probe-9002-armC-chunk8192-20260905-224845.json
08597a69bffcb9c8  bench/prefill-probe-9002-armD-dense32k-20260905-230359.json
f8f2ddc2b737f25c  bench/prefill-probe-9002-baseline-20260905-220920.json
0ed036bb9ba33ac4  bench/prefill-probe-9002-smoke-20260905-220855.json
004ecc3c12877bbf  bench/identity-verify.txt
a5b93d39e0526701  scripts/prefill-probe.py
94e941128509f319  scripts/serve-flash-next-9002.sh
72a3706da4e6ccb1  scripts/serve-flash-next-uncensored-9001.sh
```

## Instrument reconciliation — 2026-09-06 16:32 (supersedes the "~16 %" claim)

A draft of Task 5's notes in `docs/plans/2026-09-06-long-context-prefill-handoff.md`
and the commit message of `a04460d` said the ladder's serve harness and the probe
"disagree by ~16 %". They do not. The figure compared 746,8 tok/s (probe, 51.347 new
tokens, engine time) with 627 tok/s (ladder, 63.494 new tokens, client TTFT) — two
variables changed at once. Measured again with one fresh server per run, an empty
session bank, the same repo build, window 131072, chunk 2048:

| strumento | token nuovi | tempo motore | rata motore | tempo client | rata client |
|---|---|---|---|---|---|
| probe, corpo funzione-sorgente | 50.549 | 67,81 s | 745,5 | — | — |
| probe, corpo funzione-sorgente | 62.908 | 93,83 s | 670,4 | — | — |
| ladder, corpo coding-agent | 64.521 | 98,07 s | **657,9** | 98,84 s | 652,8 |

Decomposition of the 11,9 % that started this: **11,5 punti = dimensione del contesto**
(pendenza locale misurata: 6,07 tok/s ogni 1000 token in più), **0,78 punti = client vs
motore** sulla stessa identica richiesta, **0,42 punti = corpo del prompt**. Ripetibilità
del probe tra due run dello stesso tipo: 0,18 %; rumore cold conosciuto: 0,43 %. Ogni
residuo è quindi sotto il rumore: i due strumenti concordano.

Artefatti: `prefill-probe-9002-instrument-20260906-162656.json`,
`prefill-probe-9002-matched-20260906-163037.json` (raw, locali, come sopra).

## Task 3 sweep — `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT` does not move qwen4_exp prefill

2026-09-06 16:36–16:57. Serve harness, repo build, one fresh server per arm, session bank
emptied per arm, matched prompt: 62.908 new tokens, identical bytes across arms (fixed
`--tag sweep` salt), `--profile turbo --depth 3 --context-window 131072 --prefill-chunk-tokens
2048 --scheduler-mode serial`. Rate = engine `prompt_eval_time_s`.

| braccio | tok/s note |
|---|---|
| auto (16:32) | 670,4 |
| auto ctl1 | 670,2 |
| auto ctl2 | 668,5 |
| 65536 | 673,6 |
| 16384 | 669,4 |
| 32768, prima run | **637,7** |
| 32768 rip1 | 675,7 |
| 32768 rip2 | 669,0 |

`auto` mean 669,7 (n=3) vs `32768` mean 672,4 (n=2, after repeating): **+0,40 %**. Spread
across every arm excluding the first 32768 run: **1,07 %** (668,5–675,7). The 637,7 reading is
a non-replicating outlier — the reason this section records repeats: a single-arm run cannot
support a claim below ~5 %, and the draft conclusion "32768 costs 4,9 %" was already written
before the interleaved control disproved it.

Counters in **every** arm, including the most aggressive ceiling: `paged_gqa_sdpa_calls` 0,
`prefill_partitioned_paged_calls` 0, `prefill_dense_fallback_calls` 0,
`large_q_split_sdpa_fallback_calls` 0. Consistent with arm D (in-process, ceiling 32768:
267,003 s vs 267,920 s baseline) and with `profiles.py:525`, which records dense decode as the
faster side of the fence.

What the sweep could not observe: the request rows for this family carry no
`prefill_layout` / `prefill_attention_impl` keys at all, so "the lane was not chosen" is known
from absent counters, not from a positive statement of what ran. That gap is Task 3 step 1 and
it is now the critical path — see the handoff.

## Task 3 step 1 shipped, and the lane sweep that it made possible

2026-09-06 18:47–19:05. `scripts/install_repo_launcher.sh` sibling work aside, this is the
instrumentation slice: the engine row now carries `prefill_layout` and `prefill_attention_impl`.

**What was missing and why the old rows could not answer.** `PUBLIC_MTPLX_STATS_KEYS` already
listed `prefill_route`, but the request-log envelope copies a *different*, inline key set inside the
request handler, so the field never reached `request-log-<port>.jsonl`. That is why the 16:36 sweep
read four lane counters as 0: the keys were present as **dataclass defaults**, not as measurements.
The new `prefill_attention_impl` distinguishes the two by construction — `"none"` means the lane
accounting ran and no lane fired, `"unrecorded"` means the request never reached it.

Proof on the live engine (fresh repo-build server, 8k probe, one request each):

| braccio | `prefill_layout` | `prefill_attention_impl` |
|---|---|---|
| auto | `contiguous_dense_decode` | `none` |
| `MTPLX_SUSTAINED_PREFILL_LAYOUT=contiguous_then_repage` | `contiguous_then_repage` | `none` |

The field discriminates, so it reads the executed path rather than the requested one.

## Lane sweep — the ceiling changes the executed layout and costs nothing measurable

`~/.mtplx/scripts/prefill-lane-sweep.py --rounds 3`, repo build on 9003, one fresh server per arm,
session bank emptied per arm, identical prompt bytes (fixed probe tag), **51,395 tokens** per row,
9/9 rows without error. Rate = engine `prompt_eval_time_s`.

| braccio | n | media tok/s | spread intra-braccio | layout eseguito |
|---|---|---|---|---|
| auto | 3 | 744,9 | 1,61 % | `contiguous_dense_decode` |
| ceiling 32768 | 3 | 742,6 | 0,94 % | `contiguous_then_repage` |
| forced repage | 3 | 741,8 | 0,58 % | `contiguous_then_repage` |

ceiling vs auto **−0,30 %**, repage vs auto **−0,41 %**, repage vs ceiling **−0,11 %** — every delta
under the largest within-arm spread (1,61 %), so no speed effect is claimed either way. The new
result is the **why**: at this context the ceiling really does flip the executed layout
(`dense_decode` → `then_repage`), and the flip is free. `prefill_attention_impl` is `none` in all 9
rows, so neither layout engages a counted prefill lane for this family — which is the same shape of
finding as arm D's "the paged lane was never taken", now stated positively instead of inferred from
absent keys.

**Not comparable to the 16:36 numbers.** That sweep matched 62,908 new tokens; this one is 51,395.
Rates differ by ~11 % between the two context sizes, and size is a confound, exactly as in §12:
compare arms inside one sweep, never across sweeps.

## D3 / ladder in-process: the real blocker is a fresh ple cache, not a missing adapter

Measured with `build_verify_state_spec` on synthetic caches (no weights loaded):

| cache | esito |
|---|---|
| `ArraysCache(2)` | accettata, `(0, gdn, 2)` |
| `ArraysCache(4)` **appena costruita** | rifiutata: `unsupported_container:ArraysCache[partial_ple]` |
| `ArraysCache(4)` con una foglia `None` | rifiutata, stessa ragione |
| `ArraysCache(1)` / `ArraysCache(3)` | `unsupported_container:ArraysCache[1]` / `[3]` |
| `ArraysCache(2)` con foglia `None` | **accettata** |
| `FixedArraysCache(2)` (vendored) | accettata |

`qwen4_exp.make_cache()` (`models/qwen4_exp.py:5412-5422`) puts `ArraysCache(size=4)` on every
`"ple"` layer, and a new one has all four leaves `None` until the first write. So the ladder reaches
the spec builder holding caches that are *young*, not corrupt, and one such entry poisons the whole
layer list (`tests/test_graphbank_verify_state_spec.py::test_one_bad_entry_poisons_the_whole_layer_list`).
The hard failure, as opposed to an eager fallback, comes from the fixed-M4 verify lane that D4
switched on for this family. Pinning the asymmetry (a two-leaf `None` is fine, a four-leaf one is
not) is the deliberate part: the fix must populate or defer those leaves, not relax the guard and
hand the compiled core `None`.

## Task 3 step 2 — the lane seam, unit-proven: four gates, and a counter that does count

2026-09-06 22:06. `tests/test_qwen4_prefill_lane.py`, 13 cases, no weights: synthetic KV at the
pack's real head shape (`num_attention_heads 24`, `num_key_value_heads 2`, `head_dim 256`,
`full_attention_interval 4` → 12 `QSACache` layers of 48, read from the pack `config.json`).

Step 1's instrumentation had established that no counted lane fires under either sustained layout
(`prefill_attention_impl = none` on 9 of 9 sweep rows). Step 2 answers the question that
instrumentation could not: is the paged lane *unreachable*, or does it run uncounted? Answer:
unreachable on the served path, by four independent gates, in the order a request meets them.

| # | gate | site | what it does |
|---|---|---|---|
| 1 | layout scope | `generation.py` `_target_prefill_cache_layout_scope` | force-zeroes `MTPLX_VLLM_METAL_PAGED_ATTN`, `MTPLX_OWNED_ATTN_KV`, `MTPLX_BLOCK_OWNED_ATTN_KV` while *either* sustained layout is active; `auto` always resolves to one of the two, so `_make_target_prefill_cache` can never install the owned subsystem |
| 2 | wiring | `cache_state.py` `install_vllm_metal_paged_attention_kv_cache` | converts only entries exposing stock `keys`/`values`; `QSACache` keeps the KV in `.kv` beside positional indexer streams → measured `entries: 0, skipped: 1` with every env on and the inner KV already written |
| 3 | impl | `cache_state.py` `paged_attention` | the GQA route block runs only under `MTPLX_VLLM_METAL_PAGED_ATTN_IMPL` ∈ {`sdpa_2pass_paged`, `mlx_vector_paged`} and offset ≥ the 1024 two-pass threshold; the serve default sets neither |
| 4 | route window | `cache_state.py` `_paged_gqa_sdpa_route_decision_from_env` | off by default (`reason: disabled`); default window `min_q 4 … max_q 5` is decode/MTP width, so an 8192-token prefill chunk is refused `q_len_gt_max` even with the route on — no ceiling value can move this counter |

The positive half, which is what makes the zeros a measurement rather than an absence:

| condition (unit level) | observed |
|---|---|
| wired + `impl=mlx_vector_paged` + route `auto` + `q_len 4` + offset 2048, phase `prefill` | `gqa_sdpa_calls 1`, `by_route {async_per_head: 1}`, `by_phase {prefill: 1}` |
| same object, `PARTITIONED_ATTN=1`, `q_len 8192` | `partitioned_paged_calls 1`, `by_phase {prefill: 1}` — the prefill-width owned lane counts per phase |
| same object, impl unset, `q_len 4` | output correct, `gqa_sdpa_calls 0`, `route_misses {}`, `paged_attention_calls 1` — served by a lane with no counter and no miss; the one place "runs uncounted" is real, and it is *inside* the owned object |
| route decision, `(24, 24)` heads | `not_gqa` — the gate is shape-aware, not blanket |

Verdict: on the sustained serve path the owned paged subsystem is never constructed for this family,
so `paged_gqa_sdpa_calls = 0` at every rung is structural. Prefill issue criterion 5 cannot be met
by tuning on 2.11.x; the branches are client-side session size (under ~60 k, where the measured
follow-up is 1.22–1.62 s) or wiring `QSACache` into the paged install — a code change with its own
slice, because the indexer's `raw_keys`/`pooled` streams are positional buffers keyed to `kv.offset`
and must survive the conversion.

## D1 — follow-up split: the cost is prompt eval, and it is the same curve as cold prefill

2026-09-06 22:06, from `~/.mtplx/bench/prefill-probe-9001-baseline-20260905-222454.json` engine
rows (not client wall times):

| follow-up at ctx | prompt | cached | new | prompt_eval_s | ttft_s | share of ttft | ms per new token |
|---|---|---|---|---|---|---|---|
| 8 150 | 8 150 | 8 150 | **0** | 0.030 | 0.430 | 7 % | — |
| 27 399 | 27 399 | 26 720 | 679 | 1.495 | 1.516 | 99 % | 2.20 |
| 53 055 | 53 055 | 52 382 | 673 | 1.901 | 1.931 | 98 % | 2.82 |
| 104 382 | 104 382 | 103 707 | 675 | 2.835 | 2.896 | 98 % | 4.19 |

Read with the cold column of the same file (7 478 → 987 tok/s; 26 727 → 910; 52 389 → 737;
103 714 → 386; i.e. 1.014 → 2.590 ms per token, 2.56× from short to long context), this says:

- the follow-up cost is prompt eval, 98–99 % of `ttft_s` on every row that has new tokens; the only
  cheap follow-up is the one with **zero** new tokens (`cache_source ssd`), while all three paying
  rows are `cache_source ram`;
- nothing is re-prefilled beyond what is counted: `generation.py:4020-4021` sets
  `cached_tokens=restore_point` and `new_prefill_tokens=len(suffix)`, so the reported cached prefix
  *is* the restore boundary and the reported new count *is* the evaluated span. A 675-token suffix
  therefore genuinely costs 2.835 s = 238 tok/s. (This receipt first called that "1.6x worse per
  token than cold at the same context"; that compared a **marginal** rate with a **cumulative
  average**. Against the cold marginal cost of the same band the ratio is 1.065 -- see "The prompt
  phase now reports its parts" below, which is where the correction comes from.);
- a withdrawn hypothesis, recorded so it is not re-derived: an O(sessions) scan at
  `session_bank.py:1746` via `get_session` → `_prune_locked`. Neither symbol exists in that file and
  line 1746 is the boundary-true restore path. The scans that do exist (`_active_session_ids` over
  `_session_last_active.items()`, `_entries.items()` in the restore/candidate paths) are host-side
  over maps with a handful of entries — orders of magnitude too small for 2.8 s.

So D1 and D2 are one curve -- and the instrument built an hour later says so with numbers instead of
arguing from shape. The follow-up's per-token cost is 1.065x the cold **marginal** cost of the same
context band, and it does not improve when the suffix is made wider, so the cost is set by the
context a request sits at rather than by the narrow-q shape this receipt first suspected. The
breakdown that established both is the next section.

## The prompt phase now reports its parts, and the parts close D1 (2026-09-06 23:05)

Instrument: `_prompt_eval_time_from_parts` and `_prompt_eval_breakdown_is_complete` in
`mtplx/generation.py`; four new row keys (`prompt_repair_time_s`, `prompt_suffix_time_s`,
`prompt_repage_time_s`, `prompt_eval_breakdown_complete`) plus `prompt_mtp_history_time_s`, which the
request-log row was missing while the public projection already had it. One place composes the
aggregate, so a part cannot enter the total without being emitted beside it, and the completeness
flag keeps a `0.0` part from reading as "nothing happened" on a path that never decomposed -- the
same `none` versus `unrecorded` distinction the lane fields needed.
`tests/test_prompt_eval_breakdown.py` (7 cases); the 15 request-observability goldens regenerated
with **60 additions, 0 removals, 0 changed values**.

Live on the side-by-side 9003 running the repo build (9001/9002 left untouched, both 200 before and
after):

| row | cached | new | prompt_eval_s | suffix | mtp_history | repair | repage | complete |
|---|---|---|---|---|---|---|---|---|
| cold 7 847 | 0 | 7 847 | 8.034 | 0.0 | 0.2158 | 0.0 | 0.0 | **False** |
| follow-up 8 515 | 7 840 | 675 | 1.1881 | **1.1235** | 0.0342 | 0.0305 | 0.0 | **True** |
| follow-up 102 030 | 101 355 | 675 | 2.7975 | **2.7687** | 0.0 | 0.0288 | 0.0 | **True** |

The 102 k follow-up is **99.0 % suffix prefill** -- repair 1.0 %, repage nothing -- and the bank
restore (0.481 s) sits outside `prompt_eval_time_s` entirely. So the follow-up cost is not restore
bookkeeping, not the one-token repair forward, and not repaging: it is the same attention work a cold
prefill does, at the context the request sits at.

Marginal versus average, the comparison that settles it (cold rows from the 9001 baseline):

| cold rung | ms/token, cumulative average | ms/token, marginal to the previous rung |
|---|---|---|
| 7 478 | 1.0135 | -- |
| 26 727 | 1.0984 | 1.1314 |
| 52 389 | 1.3560 | 1.6244 |
| 103 714 | 2.5902 | **3.8499** |

| measurement | ms per new token | / top-band marginal |
|---|---|---|
| follow-up, 675 new at 101 355 cached (today, 9003) | 4.1018 | **1.065** |
| follow-up, 675 new at 103 707 cached (baseline, 9001) | 4.1996 | 1.091 |
| suffix-width sweep, 7 220-9 006 new at 94 208 cached | 4.448-4.853 | 1.155-1.261 |

Two conclusions:

1. **A follow-up pays the cold curve's marginal cost at its own context, within 6-9 %.** There is no
   separate restored-cache penalty to hunt, so the prefill issue's criteria 3 and 4 are one fix
   rather than two: halve the marginal cost of long-context attention and the 103 k cold prefill and
   the 103 k follow-up move together.
2. **Suffix width is not the lever.** The sweep asked for suffixes of 64 to 4 096 tokens; the bank
   matched a trimmed prefix (94 208 cached instead of 101 355), so the real suffixes were
   7 220-9 006 tokens, and per-token cost stayed flat across them. A narrow-q arithmetic-intensity
   story would have improved as the suffix widened; it did not. Recorded with its confound rather
   than re-run, because the flatness is the part that carries:
   `~/.mtplx/scripts/d1-suffix-width-sweep.py`, rows in `~/.mtplx/logs/request-log-9003.jsonl`.

Swap note bearing on criterion 7 of the prefill issue: 72.56 MiB with the 9001+9002 pair,
**139.75 MiB** while the temporary third pack (9003) was resident -- above the pair's 128 MiB bound.
The criterion is written for the pair; a third Flash-Next pack is outside its scope, and the reading
is recorded here rather than left in a shell scrollback.

## The QSA sparse prefill lane was never built on this machine — building it is the fix

2026-09-06 23:45. **Cold 103 k prefill: 251.33 s -> 104.68 s median (-58.4 % wall, +140.6 %
throughput). Follow-up at 101 k: 2.79 s -> 1.56 s median (-44.1 %). Peak 129.8 -> 114.9 GiB.** No
product code changed: a vendored native extension that ships in this tree had never been compiled
into the serving venv.

### How it was found

Task 7 step 1 went looking for the knobs inside the QSA layer, because `models/qwen4_exp.py:3760-3786`
gathers selected rows with `mx.take` and runs SDPA on the subset — so the term that grows with
context had to be the indexer's selection or the gather, not a dense attention. The layer's prefill
pipeline is gated by `_qsa_prefill_enabled()` (`:1480`), which resolves AUTO through
`qsa_prefill_lane_auto_supported()` (`:1439`). Probed directly in the serving venv:

```
nax selector disponibile      : False
prefill lane auto supported   : False
producer compile ok           : True
[mtplx] QSA sparse prefill disabled: this Metal SDK cannot compile the MPP score pipeline;
                                     using dense prefill (QSA MPP prefill scores called for an
                                     unsupported signature)
mpp compile ok                : False
_qsa_prefill_enabled()        : False
```

`qsa_prefill_lane_auto_supported` has exactly two fast consumers (`:1440-1453`): the Metal 4
TensorOps (NAX) flash kernel for M4/M5, and — for M3-class GPUs, which have no G17 tensor units —
the vendored **Steel sparse-GQA** kernel, "when it is built and probed". Neither was available, so
the model rode "the eager selector into the dense-mask reconstruction — pure tax", in the docstring's
own words. The reason the M3 consumer was missing:

```
_IMPORT_ERROR : ModuleNotFoundError("No module named 'mtplx_qsa_kernels'")
```

`native_extensions/qsa_kernels/` (vendored from oMLX PR #3244, Apache-2.0) was in the tree, with
`CMakeLists.txt`, `setup.py`, the `.metal` source and a README — never built. Its artifacts are
gitignored, so a clone does not carry them and nothing in the boot path warns that the fast lane is
off.

### The build, exactly as performed

Restore point recorded first: the venv held `nanobind 2.12.0` (dist-info present) and **no**
setuptools (uv-created venv, no pip). The extension pins `nanobind==2.15.0` exactly — a mismatch
imports cleanly and then rejects every `mx.array` (oMLX #2139) — and `mlx==0.32.2`, which the venv
already had.

```sh
uv pip install --python .venv/bin/python "nanobind==2.15.0" "setuptools>=42"
cd native_extensions/qsa_kernels && ../../.venv/bin/python setup.py build_ext --inplace
```

Two linker warnings, recorded rather than hidden: a missing framework search path
(`/Applications/Xcode_26.6.app/...`) and "building for macOS-15.0, but linking with dylib
`@rpath/libmlx.dylib` which was built for newer version 26.2". Proof per the extension's README:

```
BUILT_AGAINST_MLX     : 0.32.2      imported_mlx: 0.32.2   (receipt matches)
BUILT_AGAINST_NANOBIND: 2.15.0
preflight pipeline ok : True
lane auto supported   : True
_qsa_prefill_enabled  : True
otool -L _ext*.so     : @rpath/libmlx.dylib, @rpath/libmtplx_qsa_kernel_ops.dylib  (no build-tree path)
```

`qsa_prefill_direct.py:116-130` finds the in-place artifacts itself, so no install into site-packages
is needed. Repo stayed clean (`git status --porcelain` empty): `.so`, `.dylib`, `.metallib` and
`build/` are gitignored.

### A/B, matched, same session, side-by-side 9003

Arms differ only by `MTPLX_QSA_PREFILL` (verified present in the process environment with
`ps eww`): arm A auto (lane on), arm B `=0` (kill switch). Same pack, profile turbo, depth 3, chunk
2048, `--unique-body` so every cold row is genuinely cold. 9001/9002 untouched throughout.

| rung | arm | cold eval_s | cold tok/s | follow-up eval_s | follow-up tok/s | peak GiB |
|---|---|---|---|---|---|---|
| 52 k | B (lane off) | 68.35 | 742.0 | 1.86 | 363.3 | 117.5 |
| 52 k | **A (lane on)** | **56.24** | **902.0** | **1.47** | **460.4** | 114.9 |
| 104 k | B (lane off) | 251.33 | 403.1 | 2.79 | 241.6 | 129.8 |
| 104 k | **A (lane on)**, median of 3 | **104.68** | **968.4** | **1.56** | **431.1** | 114.9-134.9 |

Arm A at 104 k, all three runs: cold 106.46 / 104.68 / 104.50 s (spread 1.87 %), follow-up
1.56 / 1.56 / 1.56 s. Arm B reproduces the lane-off run from earlier the same evening (256.68 s /
394.9 tok/s at 22:46 vs 251.33 s / 403.1 now, 2 % apart), so the single B run is not a lucky draw.

| effect | 52 k | 104 k |
|---|---|---|
| cold throughput | +21.0 % | **+140.6 %** |
| cold wall | -17.7 % | **-58.4 %** |
| follow-up wall | -21.0 % | **-44.1 %** |
| peak memory | -2.6 GiB | -14.9 GiB |

The gain grows with context, which is the signature of removing a superlinear term: the cold marginal
cost between 50.7 k and 101.3 k falls from 3.8499 to **2.0464 ms/token**, and the follow-up still
tracks it (4.1018/3.8499 = 1.065 before, 2.3284/2.0464 = **1.138** after) — the D1 conclusion holds
on the faster curve, so criteria 3 and 4 remain one fix rather than two.

### Criterion verdicts for the prefill issue

- **Criterion 4 (cold 103 k <= 134 s): MET.** 104.68 s median of 3, against a 267.92 s baseline.
- **Criterion 3 (follow-up <= 1.40 s median of >= 3): open, but 1.56 s against a 2.805 s baseline**
  — 11.4 % above target. The curve bent; it did not flatten.
- Criterion 5 (paged lane) is untouched by this: the paged lane is still unreachable by construction,
  and the win came from a different lane entirely.

### Two defects found while measuring, both pre-existing

1. **An identical repeat request 500s.** Re-sending the same conversation (whole prompt cached, zero
   new tokens) raises `ValueError: qwen4 fixed-M4 prompt history does not match the prefetched
   cache` at `qwen4_fixed_verify.py:269`, reached from `graphbank.py:2015 install_fixed_m4`. The
   check compares the n-gram sidecar's last two tokens with the prompt tail; the sidecar is left
   advanced by the previous generation and nothing re-aligns it when there is no suffix. The check
   comes from `d6018d2` (ported from PR #391), not from this session's work. This is a retry-shaped
   request, so clients can hit it in normal use.
2. **A diverging continuation pays for a trimmed snapshot.** The stored snapshot for a 101 k prompt
   restores at **94 208** tokens, not 101 305, so a follow-up whose text diverges from the stored
   chain re-prefills ~7.7 k tokens: 36.57 s and 26.90 s measured, against 1.56 s for a continuation
   that extends the chain. Rows: `~/.mtplx/logs/request-log-9003.jsonl`, artifacts
   `~/.mtplx/bench/prefill-probe-9003-qsa{A,A2,A3,B}-*.json`.

### What is not done

9001 and 9002 are still running processes started **before** the build, so they still have the lane
off; they pick it up at their next restart. Nothing in the boot path warns when the extension is
missing, so a fresh clone silently returns to the slow curve — a doctor/wrapper check is the obvious
follow-up, and it is not built yet.

## Where the instruments behind these numbers live now (2026-09-07 00:38)

Every measurement in this receipt was produced by tools that used to exist only in the operator's
`~/.mtplx/scripts/`, outside version control. They are now in this checkout, and
`scripts/install_repo_launcher.sh` installs them byte-identical into `$MTPLX_HOME/scripts` with a
sidecar manifest (`.repo-scripts.tsv`) so drift is visible and a hand-edited copy is never
overwritten without `--force`:

| in this repo | what it produces | earlier name in the lines above |
|---|---|---|
| `scripts/prefill_probe.py` | the cold + follow-up rows per rung, read from `$MTPLX_HOME/logs/request-log-<port>.jsonl` | `~/.mtplx/scripts/prefill-probe.py` |
| `scripts/prefill_lane_sweep.py` | the three-arm interleaved lane sweep (`prefill_layout`, `prefill_attention_impl`) | `~/.mtplx/scripts/prefill-lane-sweep.py` |
| `scripts/followup_repeat.py` | the extra follow-up runs a median of >= 3 needs | `~/.mtplx/scripts/followup-repeat.py` |
| `scripts/suffix_width_sweep.py` | per-token cost against requested suffix width | `~/.mtplx/scripts/d1-suffix-width-sweep.py` |
| `scripts/prefill_ladder_baseline.sh` | the in-process ladder baseline | same name |

The historical references above keep the names they were run under; the artifacts they cite
(`~/.mtplx/bench/*.json`, `~/.mtplx/logs/request-log-*.jsonl`) are unchanged. Nothing in these five
carries a credential or a non-loopback URL, and none hardcodes an absolute path: the root is
`$MTPLX_HOME` (default `$HOME/.mtplx`) and ports, wrappers and model directories are arguments.
Checked before publishing, not assumed: `scripts/hygiene_scan.sh` exit 0 with all five present.

