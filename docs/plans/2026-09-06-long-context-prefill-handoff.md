# Long-Context Prefill Handoff — Qwen3.8 Flash-Next (qwen4_exp)

> **For agentic workers:** REQUIRED SUB-SKILL: use
> superpowers-optimized:subagent-driven-development to execute this document task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking. Every claim below carries the artifact or
> command that produced it; re-measure before you believe any absolute number.

**Goal:** Fix the four engine-side defects that made long-context prefill unusable on the
`qwen4_exp` (Qwen3.8 Flash-Next) architecture, and make them measurable in CI without a live
115 GB model load.

**Architecture:** Prefill on this arch tracks tokens already in context instead of tokens newly
processed (a 556-token turn costs 0.96 s at 6.5 k context and 2.40–2.86 s at 103 k — see
"Repetition noise" below), and no fast attention lane ever engages. The work is: (1) correct the KV
geometry constant the memory policy reasons from, (2) make the paged/QSA lane reachable and
provable-reachable through counters, (3) replace the hard 16,384-token refusal with a routed
degrade, and (4) give the prefill benchmark a serve-path mode so measurement stops being
in-process-only.

**Tech Stack:** Python 3.13, MLX/Metal, pytest, uv, the guarded `/tmp/mtplx-gpu-exclusive.lock`
convention from `docs/plans/2026-08-09-qwen35b-mtp-batch-numerics-profiles.md`.

**Assumptions:**

- Assumes the two packs named below are the only `qwen4_exp` targets that matter for this work.
- Assumes the M3 Ultra (512 GB, no M5 TensorOps path) is the reference machine; every number here
  is that machine and must be re-measured before being quoted elsewhere.
- Assumes a serve-path benchmark may be added to `prefill_bench.py`; without it, none of the
  acceptance criteria below can be tested without a 115 GB in-process load.
- Assumes `paged_gqa_sdpa_calls == 0` means "lane not used". It is read from the owned-attention
  instrument (`mtplx/generation.py:1069`), so a refactor that moves that plumbing changes the
  meaning of the counter — re-verify the plumbing before trusting the number as a test oracle.

## Version provenance — read this first

| | value |
|---|---|
| Everything measured against | installed **mtplx 2.11.0** (`~/.mtplx/bin/mtplx --version`) |
| This checkout at measurement time | **2.9.0** (`pyproject.toml`), branch `issue-308`, no `qwen4_exp` descriptor |
| After the 2026-09-06 rebase | `main` = upstream `406b5f7` (**2.11.1**), `issue-308` = `d6e4a5a` (DFlash2) rebased on it |
| Line numbers in this document | valid on the **rebased** tree, verified by grep after the rebase |

The rebase mattered: at 2.9.0 none of the symbols below existed, so an earlier draft of this
document was unusable. Re-run the anchors (`grep -n`) before patching if the tree moves again.
Residual skew: measured on 2.11.0, tree is 2.11.1 — so **Task 0 re-baselines before any fix is
judged.**

## Measured baseline (the thing to beat)

Tracked receipt: **`docs/perf/receipts/qwen38-flash-next-prefill.md`** — provenance, method, all
tables, cross-checks, and SHA-256 of every raw artifact. The raw JSONs stay local
(`~/.mtplx/bench/`; `agent-output/` is git-excluded via `.git/info/exclude`, so nothing there
reaches a clone). Reproduced below so the plan is readable without opening the receipt.

`kind` is derived from the engine's `cached_tokens` (`0` → `cold`); `prefix_hit` is one ~600-token
turn appended to the same prefix; throughput is `new_prefill_tokens / prompt_eval_time_s`.

9002 = `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`, profile turbo, depth 3, sidecar
resident (raw: `bench/prefill-probe-9002-baseline-20260905-220920.json`):

| kind | prompt | new | prompt_eval_s | new tok/s | peak |
|---|---|---|---|---|---|
| cold | 6,500 | 6,500 | 8.012 | 811 | 112.6 GiB |
| prefix_hit | 7,049 | 556 | 0.960 | 579 | 112.6 GiB |
| cold | 25,745 | 25,745 | 27.965 | 921 | 117.7 GiB |
| prefix_hit | 26,294 | 556 | 1.220 | 456 | 117.7 GiB |
| cold | 51,405 | 51,405 | 69.276 | 742 | 124.1 GiB |
| prefix_hit | 51,954 | 556 | 1.615 | 344 | 124.1 GiB |
| cold | 102,731 | 102,731 | **267.920** | **383** | 135.0 GiB |
| prefix_hit | 103,280 | 556 | **2.397** | **232** | 135.0 GiB |

9001 = `grant-ai/Qwen3.8-Flash-Next-Abliterated-MTPLX-4bit`, profile sustained, depth 1, sidecar
streamed (raw: `bench/prefill-probe-9001-baseline-20260905-222454.json`): cold 103,714 → 268.637 s
at 386 tok/s; 675-token follow-up at 104,382 → 2.835 s at 238 tok/s.

**Instrument warning, learned the hard way.** `prompt_eval_time_s` (engine) and the probe's terminal
`wall=` line (client — includes HTTP, tokenisation and teardown) differ by up to 18 %: the first
version of the table above carried the client numbers, which inflated the follow-up decay from 2.5×
to 2.9× and shifted the derived geometry. Quote the engine field, and diff any transcribed table
against the JSON before it leaves the editor.

### Repetition noise — what these numbers can and cannot prove

Four measurements of the *same* nominal work — one ~556–679-token follow-up at ~103 k context —
taken across the baseline and arms A/C/D: 2.397, 2.853, 2.757, 2.860 s → **median 2.805 s,
stdev 0.218 s, spread 17.0 %**, and the baseline's 2.397 s is the **low outlier**. Cold prefill at
the same context across three of those runs: 267.920 / 268.162 / 267.003 s → **stdev 0.611 s,
spread 0.43 %.**

Consequences that bind everything below:

- Cold-path claims are solid at ±0.5 %: the profile, window, chunk and ceiling results all stand.
- Follow-up claims are **not** resolvable below ~20 % from single runs. "Follow-up unchanged" is
  true for A/C/D against each other (2.757–2.860 s) and false against the baseline's 2.397 s. The
  honest decay against the 0.960 s at 6.5 k is **2.50×–2.98×**, and the single 2.5× figure that
  earlier drafts of this document (and its receipt) quoted was taken from the fastest of four.
- Acceptance criterion 2 therefore requires ≥3 repetitions judged on the median.

The two reported symptoms are both in that table: a 103 k cold start costs 268 s, and the same
556-token turn costs 0.96 s at 6.5 k context and 2.40 s at 103 k (2.5×). Every `prompt_eval_s` here
is the engine field, never the client wall time.

## Defects to fix

### D1 — the 16,384-token guard returns HTTP 500 on any non-sustained profile

- `mtplx/generation.py:1205` `_unsafe_long_context_prefill_guard_tokens()` defaults to **16384**;
  `:1219` `_assert_safe_long_context_prefill()` raises `RuntimeError("Blocked unsafe long-context
  MTP prefill path: …")`; called at `:4957`.
- `mtplx/profiles.py`: `SUSTAINED_PREFILL_ENV` at `:517`; `stable` at `:734` and
  `performance-cold` do **not** merge it, `sustained` and `turbo` (`:787`) do.
- Observed: 9001 on its own contract's `recommended_profile: stable` answered 500 at **26,727
  prompt tokens**; after switching to `sustained`, the identical request measured 915.5 tok/s.
- Consequence: a model pack can ship a runtime contract whose recommended profile cannot serve
  agent-sized prompts at all. The guard should route to the sustained prefill path, or the
  contract validator should reject the profile at startup — not fail per request.

### D2 — prefill tracks retained context; no fast lane ever engages

- `paged_gqa_sdpa_calls`, `prefill_partitioned_paged_calls`, `prefill_dense_fallback_calls`,
  `attention_dense_fallback_calls` all read **0** at every rung on both packs.
- `prefill_attention_impl` and `prefill_layout` are absent from the request-log rows entirely.
- 262,144 is advertised by the packs' `mtplx_runtime.json` and by
  `context_window_policy` in `/mtplx/settings`, so users will hit the tail.

### D3 — the dense-decode ceiling reasons from a KV constant 3.25–3.81× low

- `mtplx/generation.py:1499` hard-codes the default `MTPLX_DENSE_KV_BYTES_PER_TOKEN` to **65536**;
  `:1502` applies `MTPLX_DENSE_DECODE_RAM_PERCENT` default **15**; `:1533` emits the serve-log
  line `dense-decode ceiling auto: 262144 tokens (15% RAM over 65536 B/token — MODEL DEFAULT, set
  MTPLX_DENSE_KV_BYTES_PER_TOKEN for non-Qwen3.8 geometry)`. Other sites: `memory_plan.py:396`,
  `server/openai.py:3313`, `:3414`, `:3422`.
- Measured marginal from `peak_memory_bytes` across rungs: **249,526 B/token (244 KiB)** on the
  resident-sidecar port and **213,309 B/token (208 KiB)** on the streamed one.
- The engine's own refusal message agrees on magnitude: KV on **12 of 48 layers at ~24 KB/token**
  (≈288 KB/token).
- Corrected arithmetic: 262,144 needs **172.0 GiB** (9002) / **136.0 GiB** (9001) against the
  192.0 GiB engine budget — reachable, ~20.0 GiB of headroom on 9002. And at the corrected constant
  15 % of 512 GiB still permits **330,480 tokens** (9001's geometry: 386,591), above the 262,144
  window, so **correcting the constant does not by itself make the paged lane fire** — that is
  D2's job.

### D4 — `bench prefill-ladder` cannot run this arch, and has no serve-path mode

- Crash: `prefill_bench.py:850 run_prefill_ladder` → `generate_mtpk` → `graphbank`
  `forward_ar_capture` → `_fallback` → `AttributeError: 'DecoderLayer' object has no attribute
  'input_layernorm'`. Preceded by `compiled-verify prewarm {"skipped":
  ["promotion_failure:empty_kv_cache"], "complete": false}`, produced at
  `mtplx/graphbank.py:957`.
- The unguarded attribute appears at `mtplx/gdn_capture.py:2892`, `:2948`, `:3033` (also
  `mtp_patch.py:905`, `laguna_compiled_step.py:677`, `benchmarks/runners/verify_profile.py:206`).
  `qwen4_exp` layers do not carry that attribute name.
- `prefill_bench.py` has no url/port/harness handling, so the ladder can only load in-process;
  `rt = load(getattr(args, "model"), mtp=True)` at `prefill_bench.py:1108`. Result: measuring the
  serve path needed an external script (`~/.mtplx/scripts/prefill-probe.py`), which is a symptom.

### Status of D4 after 2026-09-06 (measured on the repo runtime, not theorised)

The `AttributeError` is gone: `runtime.forward_ar_capture` now asks
`gdn_capture.generic_hybrid_capture_blocker` first and raises a named error that says
which attribute is missing, which `model_type` it hit, and which lane to install; and
`prefill_bench._apply_family_verify_lane_override` sets `MTPLX_QWEN4_FIXED_M4_VERIFY=1`
for `qwen4_exp` packs the way the server does (commit `b742b8c`, tests in
`tests/test_qwen4_capture_lane.py`, 8 cases).

**The ladder still cannot run qwen4_exp.** With the lane now enabled it stops one step
later, at the lane's own validator (`graphbank.py:1203`):

```
RuntimeError: qwen4 fixed-M4 verifier refused: unsupported_container:ArraysCache[...]
```

because the ladder's `runtime.load(...)` builds `ArraysCache` where the family lane
requires the server's owned-container caches. That is Task 5 Step 2's argument stated as
a fact: hand-mirroring server env in the bench reaches a second wall, so the ladder needs
to drive a real server (`--url/--port`) instead of approximating one. Task 5 Step 1 is
done; Step 2 is now the critical path, and Task 3 (D2) depends on it.

### D5 — KV quantization is refused for this family

- `mtplx/backends/descriptors.py:481` carries the refusal text ("…attention has no validated
  quantized-cache lane yet."); the server printed the Flash-Next-specific refusal and downgraded
  `q8` → `off` silently at load. `:147` is the generic `kv_quant_unsupported_reason` string.
- Worth deciding: silent downgrade is fine as a default, but the served `/mtplx/settings`
  response should report the *effective* mode, not the requested one.

### D6 — test isolation: `test_public_cli.py` leaks into `test_generation_sustained.py`

- Reproduced: `pytest tests/test_generation_sustained.py` alone → green. Same file **after**
  `tests/test_public_cli.py` → **6 failures**:
  `test_lazy_bonus_verify_shortens_full_accept_verify_input`,
  `test_lazy_target_distributions_inline_bonus_avoids_bonus_reforward`,
  `test_lazy_target_distributions_stop_after_first_rejection`,
  `test_lazy_bonus_verify_skips_d1_by_default`,
  `test_omit_speculative_bonus_skips_bonus_distribution_row`,
  `test_trim_commit_keeps_rejected_verify_prefix_without_reforward` — with
  `assert [1, 4] == [1, 3, 1]` at `tests/test_generation_sustained.py:716` and IndexErrors at
  `:747`, `:777`, `:845`.
- **Verified pre-existing**: identical failures from an isolated `main` worktree at `406b5f7`, so
  it is not a rebase artifact.
- Also: tests asserting `DEFAULT_HF_MODEL_ID` (`test_public_cli.py::test_tune_default_dry_run_is_not_legacy_models_path`,
  `::test_quickstart_default_missing_cache_is_not_legacy_models_path`) read the **real**
  `~/.mtplx/config.toml` and fail on any machine whose default model differs. They pass under
  `HOME=$tmp MTPLX_HOME=$tmp/.mtplx`. Pin the environment in those tests.

  *Status 2026-09-06: fixed.* Both halves — see Task 1 Steps 3 and 4 for the mechanism and the
  narrower `MTPLX_CONFIG` pin that replaced the `HOME` idea.

## Levers already eliminated — do not re-test

| lever | result | evidence |
|---|---|---|
| profile turbo vs sustained | no effect on prefill (267.92 s vs 268.64 s at 103 k) | the two baselines above |
| `--context-window` 262,144 → 131,072 | 0.09 % on cold (268.162 s), follow-up within its 17 % noise band; saves 11.7 GiB | arm A |
| `--paged-kv-quantization q8` | refused by the engine, downgraded to `off` | D5 |
| `--prefill-chunk-tokens` 8192 | **17.9 % slower** (315.868 s) and +22.7 GiB | arm C |
| `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=32768` | −0.34 % (267.003 s); the auto ceiling line disappears from the log when set explicitly, so the override was read | arm D |

MTPLX's own comment at `mtplx/profiles.py:525` records dense decode as the *faster* side of that
fence ("decode cliff: 12.0 -> 18.44 tok/s once dense decode holds"), and `:530` sets
`MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT: "auto"`. That is consistent with arm D being flat rather
than better: anything that forces repaging low is expected to lose.

## File structure

- Modify `mtplx/generation.py` — guard at `:1205-1219`, call at `:4957`, counter plumbing at
  `:1069`, KV geometry at `:1499` and `:1502`, serve-log line at `:1533`.
- Modify `mtplx/profiles.py` — `SUSTAINED_PREFILL_ENV` (`:517`), the ceiling default (`:530`), and
  the profile table (`:734` stable, `:787` turbo).
- Modify `mtplx/memory_plan.py` (`:396`) and `mtplx/server/openai.py` (`:3313`, `:3414`, `:3422`) —
  the ceiling consumers and the `Memory plan` line.
- Modify `mtplx/gdn_capture.py` — unguarded `layer.input_layernorm` at `:2892`, `:2948`, `:3033`.
- Modify `mtplx/backends/descriptors.py` — `qwen4_exp` family at `:499`, `:565`, `:585`, `:1249-1254`,
  `:1306`, `:1343`; KV-quant refusal text at `:481`.
- Modify `mtplx/graphbank.py` — `failures["empty_kv_cache"]` at `:957`, the promotion failure that
  preceded the ladder crash.
- Modify `mtplx/prefill_bench.py` — add a serve-path harness (`run_prefill_ladder` at `:850`).
- Modify `tests/test_generation_sustained.py`, `tests/test_public_cli.py` — environment isolation
  (D6).
- Create `tests/test_qwen4_prefill_lane.py` — the counter oracle for D2.
- Create `tests/test_long_context_profile_guard.py` — D1's startup-validation behaviour.

## Known environment state — read before judging any test run here

This checkout's `.venv` was built for 2.9.0 and the rebase moved the tree to 2.11.1, whose
`pyproject.toml:33` floor is `mlx>=0.32.2` while the venv carries **MLX 0.32.0**. Consequences,
reproduced identically with and without the D6 fix (worktree at the pre-fix commit: 7 failures both
ways):

- `tests/test_release_pins.py::test_installed_mlx_meets_the_pyproject_floor` fails by design — its
  own docstring calls it a venv-drift canary.
- 6 failures in `tests/test_graphbank_compiled_verify.py` (`CompiledVerifyParityError: compiled
  verify parity mismatch`, `assert 4 == 0`, `assert 'hidden' == 'logits'`) — same drift, compiled vs
  eager on an older MLX.

A full `pytest tests/` here therefore ends at **7 failed / ~6,900 passed**, and that is an
environment fact, not a code regression. **Task 0 Step 2 must build a fresh venv resolved against
the 2.11.1 pins** (`uv sync`, or `pip install -e ".[dev,server]"` in a new environment) instead of
reusing this one — otherwise every arm is measured on a runtime the tree does not support, and these
parity failures will be read as regressions caused by the fix.

**Superseded 2026-09-06 22:18 — that baseline no longer holds, and one flake surfaced on the way.**
The seven failures above are fixed in-tree: the compiled-verify parity contract is now explicit
instead of bit-exact-by-assumption (`baa6182`), and the stale source assertion in
`test_a3b_compiled_target_prefix.py` was corrected in the same commit. A full `pytest tests/` on
this venv then ran **6 620 tests across 371 files, exit 0, zero `FAILED`** — measured 2026-09-06
20:44 at `98d5c52` and again 21:28 at `16ec337`. Task 0 Step 2's fresh-venv requirement still stands
for *measurements* (this venv's MLX drift is what produced the parity errors); it is no longer what
stands between you and a readable suite result.

The 22:10 run then failed exactly one test. It is recorded here because it is a fixture defect that
can hit any full run, not because it belongs to this plan's subject:

- `tests/test_memory_pressure_guard.py::test_bank_shrink_to_bytes_evicts_lru_first` → `assert 1 == 2`.
- Not a code regression: the test passes in isolation and so does its whole file; `total_nbytes` is
  a plain `sum(entry.nbytes ...)` (`session_bank.py:633`) and `_evict_entry` pops one entry with an
  identity fallback (`:2944-2950`), so nothing cascades that could stop the loop early.
- Mechanism: the fixture built its keys as `token_ids=(hash(name) % 1000, 2, 3)`, and `str` hashing
  is salted per process. When two of the three names collide mod 1000 the fabricated table holds
  **2** entries instead of 3, `shrink_to_bytes(500)` correctly stops after one eviction, and the
  assertion fails `1 == 2`. Reproduced on demand by giving two entries the same key (3 distinct keys
  → `evicted 2`; colliding keys → `evicted 1`); 12 fresh processes showed the collision is rare
  rather than systematic, which is why four earlier full runs were green.
- Fix: deterministic literal keys in both tests that used the pattern — the sibling
  `test_bank_shrink_protect_active_never_evicts_the_live_session` carried the same fixture and the
  same latent flake. Assertions unchanged, nothing skipped or loosened; 14/14 fresh-process runs of
  the file green afterwards.

## Tasks

### Task 0 — Re-baseline on this tree before touching anything

- [ ] **Step 1:** acquire `/tmp/mtplx-gpu-exclusive.lock`; stop the user's live 9001/9002 servers
      first (`mtplx stop --port 9001`, `--port 9002`) — they hold ~115 GB each and the ladder loads
      a second copy in-process.
- [ ] **Step 2:** install this checkout into a scratch venv (`python -m pip install -e ".[dev,server]"`)
      and confirm `mtplx --version` reports 2.11.1, not the 2.11.0 the numbers above came from.
- [ ] **Step 3:** run the 8-rung probe on both packs, **repeating the 103 k rung three times**
      (cold spread is 0.43 %, but follow-up spread is 17 % and the single-shot baseline turned out
      to be the fast outlier). Store results next to the 2.11.0 baseline. A median cold `> 400 s` or
      median follow-up `> 3.3 s` at 103 k means the tree regressed, not that a fix failed.
- [ ] **Step 4:** commit nothing; append the two new artifact hashes to
      `docs/perf/receipts/qwen38-flash-next-prefill.md` and record there whether 2.11.1 moved the
      103 k numbers. If it did, every target in "Acceptance criteria" must be restated from the new
      baseline, not from the 2.11.0 figures.

### Task 1 — D6 first (it unblocks honest CI signal) — **DONE 2026-09-06**

- [x] **Step 1:** regression test written as `tests/test_suite_env_isolation.py` — four
      order-pinned tests asserting (a) a raw `os.environ` write is visible within its own test,
      (b) it is gone in the next test, (c) `apply_profile_env("stable")` really does write the
      process env (so this file's premise fails loudly if the CLI path ever stops mutating it), and
      (d) the six measured leak keys are absent afterwards. Chosen over "run file A then file B and
      assert green", which cannot be expressed as one test.
- [x] **Step 2:** red captured before the fix — `test_b` failed on the sentinel, `test_d` on
      `MTPLX_DROP_EVENTS`, `test_a`/`test_c` passed; the file pair run failed 6 tests.
- [x] **Step 3:** leak named: `mtplx/profiles.py:953 apply_profile_env()` writes a whole profile
      dict into `os.environ` when no mapping is passed — by design, that is how the daemon child
      inherits it — and the one-shot CLI paths never re-apply the `previous` map it returns. Keys
      observed: `MTPLX_BATCH_TARGET_ARRAYS`, `MTPLX_DROP_EVENTS`, `MTPLX_LAZY_MTP_HISTORY_APPEND`,
      `MTPLX_LAZY_TARGET_DISTRIBUTIONS`, `MTPLX_LAZY_VERIFY_LOGITS`,
      `MTPLX_SKIP_VERIFY_SNAPSHOT`. Fix is suite-side (product behaviour is right for a short-lived
      process): `tests/conftest.py::_hermetic_mtplx_state` now snapshots `os.environ` after its own
      `setenv` calls and restores it on teardown, covering the raw writes `monkeypatch` cannot see.
      Not a reorder.
      Caveat for whoever revisits this: the leak needs the **whole** `test_public_cli.py` collection
      to appear — that one one-shot test paired with `test_generation_sustained.py` stays green, so
      verify on the file pair, not on a single test id.
- [x] **Step 4:** pinned `MTPLX_CONFIG` to the scratch dir rather than `HOME`/`MTPLX_HOME` — the
      narrower canary (`mtplx/config.py:114` already honours it), and it fixes the two
      `DEFAULT_HF_MODEL_ID` tests on a machine whose live config names a Flash-Next pack. Verified by
      running them against the untouched `~/.mtplx/config.toml`.
- [x] **Step 5:** pair green (72 passed, 1 skipped); new isolation file green; full `tests/` run
      executed — read "Known environment state" before interpreting its 7 failures.

### Task 2 — D1 guard: degrade instead of 500

- [ ] **Step 1:** failing test — `stable` profile + 17,000-token prompt must produce a completion,
      not `RuntimeError`; and the response/telemetry must say which lane served it.
- [ ] **Step 2:** red.
- [ ] **Step 3:** implement: at `:4957` route to the sustained prefill lane instead of raising, or
      validate `recommended_profile` against the guard at load and refuse *startup* with the same
      actionable text. Pick one; do not keep both behaviours.
- [ ] **Step 4:** add `tests/test_long_context_profile_guard.py`; also assert the pack-contract
      validator rejects `recommended_profile` values that cannot serve their own advertised
      `context_length`.
- [ ] **Step 5:** commit.

### Task 3 — D2 paged lane: step 3 executed (negative), step 1 is now the critical path

Step 3 ran 2026-09-06 from the serve harness: `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT` ∈
{auto, 65536, 32768, 16384} at a matched 62.908-new-token prompt, one fresh server per arm.
Result: **+0,40 %** between `auto` and `32768` means, 1,07 % spread across arms, and
`paged_gqa_sdpa_calls` **0 in every arm** including the aggressive ceilings. The ceiling knob
does not reach this family's prefill, so "make it fire" cannot be answered by tuning.
Table, run list and the one non-replicating outlier (a first 32768 reading of 637,7 tok/s that
repeats put at 675,7 / 669,0): `docs/perf/receipts/qwen38-flash-next-prefill.md`,
"Task 3 sweep".

That leaves step 1 as the only way forward, and it is a prerequisite rather than a nicety:
qwen4_exp request rows carry no `prefill_layout` / `prefill_attention_impl` at all, so today we
know the lane did not run only from absent counters. Implement the emission before deciding
whether the lane is unreachable, or every further sweep stays unfalsifiable.

### Task 3 — D2 paged lane: make it fire, then make it provable

- [x] **Step 1:** instrument first — emit `prefill_attention_impl` and `prefill_layout` on every
      request row, not just chunked ones (they are absent today, which made this hunt slow).
      **Done 2026-09-06.** `generation.py` derives the lane name from the prefill-phase counters
      (`derive_prefill_attention_impl`) and copies the executed layout into `prefill_layout`;
      `openai.py`'s envelope key set — hoisted to `REQUEST_ENVELOPE_LANE_KEYS` so a dropped key
      fails a unit test instead of surfacing only after a 115 GB boot — carries both to the
      request-log row. Proof: live 8k probe reads `contiguous_dense_decode` vs
      `contiguous_then_repage` across two arms (`tests/test_prefill_attention_impl.py`, 9 cases),
      and the field distinguishes `"none"` (accounting ran, no lane fired) from `"unrecorded"`
      (never reached it) — the distinction the 16:36 sweep lacked when it read four counters as 0.
- [x] **Step 2 — DONE 2026-09-06, and the seam exists: the lane is unreachable by construction on
      the served path.** `tests/test_qwen4_prefill_lane.py` (13 cases) pins four independent gates
      in the order a request meets them:
      1. `_target_prefill_cache_layout_scope` (`generation.py`) force-zeroes
         `MTPLX_VLLM_METAL_PAGED_ATTN` / `MTPLX_OWNED_ATTN_KV` / `MTPLX_BLOCK_OWNED_ATTN_KV` while
         *either* sustained layout is active — and `auto` always resolves to one of the two — so
         `_make_target_prefill_cache` can never install the owned subsystem.
      2. `install_vllm_metal_paged_attention_kv_cache` converts only entries exposing stock
         `keys`/`values`. This family's 12 full-attention layers carry `QSACache` (KV inside `.kv`,
         beside positional indexer streams), so they are skipped with every env on and the inner KV
         already written: measured `entries: 0, skipped: 1`.
      3. The GQA route block inside `paged_attention` runs only under
         `MTPLX_VLLM_METAL_PAGED_ATTN_IMPL` in {`sdpa_2pass_paged`, `mlx_vector_paged`} with offset
         past the 1024 two-pass threshold; the serve default sets neither.
      4. The route is off by default and its q window is decode/MTP width (`min_q` 4, `max_q` 5):
         an 8192-token prefill chunk is refused `q_len_gt_max` even with the route on, so no
         ceiling sweep can ever move this counter.

      The positive half is what turns the zero into a measurement rather than an absence: with the
      object wired, paged impl, decode-width q and the pack's real head shape (24 query / 2 KV /
      head_dim 256), `gqa_sdpa_calls = 1` with `by_route {async_per_head: 1}` and
      `by_phase {prefill: 1}`; a prefill-width chunk with `PARTITIONED_ATTN` on gives
      `partitioned_paged_calls_by_phase {prefill: 1}`. And with a non-paged impl the same call is
      served with **no** counter and **no** recorded miss — the one place "runs uncounted" is real,
      and it is inside the owned object, not on the serve path.

      Verdict for criterion 5 of the prefill issue: not satisfiable by tuning on 2.11.x. Either the
      client-side branch the criterion already names (sessions under ~60 k), or wiring `QSACache`
      into the paged install — its own slice, since the indexer's raw/pooled streams are positional
      buffers keyed to `kv.offset` and must survive the conversion.
- [x] **Step 3:** sweep the lane from the *serve* harness, not the in-process one. **Partly done
      2026-09-06, and re-scoped:** `~/.mtplx/scripts/prefill-lane-sweep.py` ran three arms
      (auto / ceiling 32768 / forced repage) × three interleaved rounds at **51,395** tokens on a
      fresh server with an emptied bank. Result in the receipt: deltas −0,30 % / −0,41 % / −0,11 %
      against a 1,61 % within-arm spread — no speed effect — and the ceiling is now *seen* flipping
      the executed layout, which the four-value 62,908-token version could not show. The
      16,384/65,536 values were not re-run; they are unnecessary unless step 2 shows a lane that
      could care.
- [x] **Step 4:** commit the sweep table even if the result is negative. **Done** with the step 1
      commit (receipt sections above).

### Ladder in-process (Task 5 step 3) — the blocker is a fresh `ple` cache, not a missing adapter

Measured on synthetic caches, no weights: `build_verify_state_spec` accepts `ArraysCache(2)` but
rejects a **newly constructed** `ArraysCache(4)` as
`unsupported_container:ArraysCache[partial_ple]`, because all four leaves are `None` until the
first write. `qwen4_exp.make_cache()` (`models/qwen4_exp.py:5412-5422`) puts a size-4 cache on
every `"ple"` layer, and one such entry poisons the whole layer list. Pinned in
`tests/test_graphbank_verify_state_spec.py` (7 cases), including the asymmetry that a two-leaf
`None` *is* accepted — which is why the fix must populate or defer the leaves rather than relax
the guard and hand the compiled core `None`.

- [x] **Confirm by run, and the run said the proposed A/B was the wrong experiment.** With the
      lane at 0 the ladder dies earlier, in `runtime.forward_ar_capture`'s named RuntimeError (the
      D4 guard), so `=0` could not isolate anything — reading the code first showed that, and the
      run that mattered was the default one. 2026-09-06 19:1x, 9001/9002 stopped, ladder at 2k
      context, 20 s, both servers restarted to health 200 afterwards:

      ```
      compiled-verify prewarm {"skipped": ["unsupported_container:ArraysCache"], "complete": false}
      ...
      graphbank.py:3548, in _fallback
          raise RuntimeError(f"qwen4 fixed-M4 verifier refused: {reason}")
      RuntimeError: qwen4 fixed-M4 verifier refused: unsupported_container:ArraysCache
      ```

- [x] **Root cause 1 — class identity, fixed here.** The reason string has no `[N]` suffix, so it
      came from the *final* branch (`unsupported_container:{type(entry).__name__}`): `isinstance`
      failed on an object whose class is named `ArraysCache`. `a3b_mtp_batch:38` installs the
      vendored cache fix **at import time**, which rebinds `mlx_lm.models.cache.ArraysCache`, while
      `qwen4_exp.py:53` had already frozen the stock object with `from ... import ArraysCache`.
      Two different class objects, one process. Reproduced and pinned in ~10 lines with no weights
      (`tests/test_qwen4_arrays_cache_identity.py`): before the fix
      `build_verify_state_spec([q4.ArraysCache(2)]) -> (None, 'unsupported_container:ArraysCache')`.
      `qwen4_exp` now resolves the class at call time. Side effect worth noting: until this change,
      this family's GDN caches were **stock** `ArraysCache` instances in the ladder path, i.e.
      without the deferred-advance bookkeeping that the vendored class exists to add.
- [x] **Root cause 2 — CLOSED as not a blocker; the prediction below was wrong and stays on
      record.** By the time the ladder builds the spec, the `ple` leaves are already written: a probe
      over the real pack reported the spec ACCEPTED with `entries=48 total None leaves=0`, including
      the size-4 cache at layer 1, and the ladder's remaining failure was the verify-strategy
      mismatch fixed in `98d5c52`. So this never gated the ladder. With identity fixed, the same 48-layer
      cache list (36 GDN, one of them `ple`, 12 QSA) is now refused for the *other* reason:
      `unsupported_container:ArraysCache[partial_ple]` — `make_cache()` gives `ple` layers a
      size-4 cache whose leaves are all `None` until the first write, and the guard requires four
      real leaves. The options are populate-or-defer those leaves before the spec is built, or
      teach the fixed-M4 core an unwritten `ple` slot. Relaxing the guard is not an option: it
      would hand the compiled verifier `None` where it expects arrays, and the two-leaf case
      already accepts a `None` slot (`tests/test_graphbank_verify_state_spec.py` pins the
      asymmetry). **Standing instruction, supersedes the wording above:** nothing in this tree gets
      routed to the upstream author. Their PRs sit unreviewed for months, so the fork owns and
      documents these fixes locally. Do not propose filing issues or PRs upstream.


### Task 4 — D3 geometry constant per architecture

- [ ] **Step 1:** derive `bytes/token` from the model config (KV-bearing layers × heads × dims ×
      dtype) instead of the 65,536 MODEL DEFAULT; `descriptors.py` already knows `qwen4_exp` is
      12 KV layers of 48.
- [ ] **Step 2:** failing test asserting the announced ceiling for a `qwen4_exp`-shaped config
      matches the derived value, and that an unset override still logs the derived number.
- [ ] **Step 3:** implement; keep `MTPLX_DENSE_KV_BYTES_PER_TOKEN` as the escape hatch and log
      which of the two won.
- [ ] **Step 4:** re-run the 8-rung probe; expect **no** throughput change from this task alone
      (see D2 note) — if it does change, something else moved and must be explained.
- [ ] **Step 5:** commit.

### Task 5 — D4 benchmark: serve-path harness + the norm crash — **steps 1 and 2 DONE 2026-09-06**

Steps 1 and 2 shipped as `b742b8c` and `a04460d`. The instrument question Step 2 raised is
closed; see "Instrument reconciliation" in the receipt. Measured against the live repo
build, the ladder runs qwen4_exp for the first time:

```
mtplx bench prefill-ladder --harness direct-http --port 9002 \
  --model ~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \
  --profile turbo --contexts 32768,65536 --max-tokens 16 --json
```

Three facts this harness exposed that the in-process ladder could not:

- **Cold rows need a fresh server.** The session bank survives requests, so a repeated
  context answers `cached_tokens == prompt_tokens` with a 0 tok/s prefill (measured:
  32777/32777, ttft 0.30 s). Recorded in `serve_harness_note`; Task 3 must not read
  such a row as a result.
- **Instrument disagreement was a measurement error of mine, now closed.** The 16 %
  quoted in `a04460d` compared 746,8 tok/s (probe, 51.347 new tokens, engine time) with
  627 tok/s (ladder, 63.494 new tokens, client TTFT) — size and measurement point changed
  together. Re-measured on one fresh server per run with an empty session bank: probe
  670,4 tok/s at 62.908 new, ladder 657,9 at 64.521 new; normalising the probe to the
  ladder's size gives 660,7, so the prompt body accounts for **0,42 %**, client-vs-engine
  overhead for **0,78 %**, and the rest was context-size decay (6,07 tok/s per 1000
  tokens). All under the measured 0,43 % cold noise. Full table:
  `docs/perf/receipts/qwen38-flash-next-prefill.md`, "Instrument reconciliation".

Step 3 stays open: the ladder's rows now carry `cached_tokens` / `new_prefill_tokens`,
which is the probe's substance, but there is no `kind` label yet, and the probe remains
the only tool that can drive *my own* wrappers' ports with a per-port session bank.



- [x] **Step 1 (shipped 2026-09-06, differently from written):** the defect is real and the seam is
      clean, but guarding three attribute lookups would have left the ladder on a lane the product
      never runs. What shipped instead: `generic_hybrid_capture_blocker()` + a named `RuntimeError`
      in `runtime.forward_ar_capture` (`b742b8c`), the family verify-lane override in
      `prefill_bench` (`b742b8c`), then the actual unblock -- reusing the server's own family rule so
      qwen4_exp verifies `batched` and rides `MTPLX_FAMILY_CAPTURE_COMMIT` (`98d5c52`,
      `tests/test_qwen4_family_verify_strategy.py`, `tests/test_qwen4_capture_lane.py`). Live
      result: the ladder completes 2048 and 131072 in one run.
- [x] **Step 2 (shipped as `--harness direct-http`, `a04460d`):** no new flag was invented -- the
      existing `--url/--port` pair selects the harness, so the ladder measures a live server. The
      noise condition held (the harness reproduced the probe's rate on the same prompt inside the
      measured 0.43 % cold spread) and it exposed two facts that changed the plan: the session bank
      survives `mtplx stop`, so a cold row needs a fresh server or an emptied bank; and the engine
      will not answer `gen_config` unless the served id matches `--model-id`, which is why the
      side-by-side port had to name itself `mtplx-flash-next`.
- [x] **Step 3 — closed by decision on 2026-09-07, and the decision is NOT the porting this step
      asked for.** The row-kind logic was never moved into `prefill_bench`, and it should not be:
      what made the external tool a liability was that it lived outside version control, not that it
      was a second program. As of 2026-09-07 all five instruments are in this checkout
      (`scripts/prefill_probe.py`, `prefill_lane_sweep.py`, `followup_repeat.py`,
      `suffix_width_sweep.py`, `prefill_ladder_baseline.sh`) and `install_repo_launcher.sh` installs
      them byte-identical into `$MTPLX_HOME/scripts` with a sidecar manifest, so the ladder and the
      probe are versioned siblings reviewed together instead of a repo tool plus an unversioned one.
      The two stay deliberately separate on purpose: the ladder measures in-process, the probe reads
      the engine rows a live server emits, and the receipt's numbers need both. If someone later
      ports the row-kind derivation anyway, this box is where the reason not to is recorded.
- [x] **Step 4:** committed with the harness work (`a04460d`) and the receipt updates (`98f8604`).

### Task 6 — D5 honest reporting

- [ ] **Step 1:** failing test: request `q8` on a family that refuses it → `/mtplx/settings`
      reports effective `off` plus the refusal reason, not the requested value.
- [ ] **Step 2:** implement in `descriptors.py` / the settings projection; commit.

## Acceptance criteria

1. 103 k cold prefill ≤ **134 s** (baseline 267.920 s / 383 tok/s → the 2.0× line). One run is
   enough: cold-path spread measured 0.43 %.
2. 556-token follow-up at 103 k ≤ **1.40 s**, as the **median of ≥3 runs** (baseline median
   2.805 s → the 2.0× line). A single run cannot satisfy this: follow-up spread is 17.0 %, and the
   2.397 s published in the baseline table is the fast outlier of four measurements.
3. `paged_gqa_sdpa_calls` non-zero at ≥100 k **or** a written conclusion in `docs/perf/` that the
   lane is unreachable for `qwen4_exp`. No silent third option.
4. A pack whose `recommended_profile` cannot serve its own advertised `context_length` fails at
   startup, not at request 3.
5. `pytest tests/test_public_cli.py tests/test_generation_sustained.py -q` green in one process.
6. Every number claimed here reproduced on 2.11.1 by Task 0, with both JSON paths recorded in
   this document.
7. `vm.swapusage` stays `used = 0` during the run. Caveat from the field: with both Flash-Next
   ports plus an unrelated 161 GiB Metal server resident, swap went non-zero (81.56 MiB) — so this
   criterion is only meaningful under the exclusive GPU lock with one model loaded.

## Repo-workflow notes learned the hard way

- `upstream` is a **blobless partial clone** (`partialCloneFilter = blob:none`). A plain
  `git rebase main` on this box issued hundreds of one-object promisor fetches (visible with
  `GIT_TRACE=1` as `fetch upstream … --filter=blob:none --stdin`) and never finished in 15 min.
  Two fixes, both verified: hydrate the trees involved
  (`git ls-tree -r <rev> | awk '$2=="blob"{print $3}' | git cat-file --batch-check`) and use the
  explicit `git rebase --onto main <old-parent> issue-308` form, which skips the patch-id
  symmetric-difference that triggers the storm.
- Diagnose with `GIT_NO_LAZY_FETCH=1` — it turns the silent stall into an instant
  `unable to read <oid>`.
- `origin/main` does not exist locally; `main` tracks `upstream/main` (youssofal/MTPLX) while
  feature branches track `origin` (Deviad/MTPLX). `git pull --ff-only` on `main` fails by design
  because upstream rewrote history (440 local-only vs 967 upstream-only at the time of the
  rebase); `main` was fast-forwarded by `git reset --hard upstream/main` with
  `backup/pre-update-main` kept at the old tip, and `backup/pre-rebase-issue-308` at the
  pre-rebase `d0518c9`. **Nothing has been pushed.**
- Run tests with a pinned empty `HOME`/`MTPLX_HOME` or they will read the operator's live config
  (this is what produced two spurious failures during the rebase verification).

### Task 7 — the long-context term lives inside the QSA layer: attribute it before wiring anything

Written 2026-09-06 23:15, after Task 3 step 2 (the paged lane is unreachable by construction) and
after the prompt-phase breakdown (a follow-up is 99.0 % suffix prefill and costs 1.065x the cold
marginal rate of its band). This is the branch criterion 5 of the prefill issue names, and it is the
only remaining engine-side lever for criteria 3 and 4.

**The trap this task is sequenced to avoid.** Wiring `QSACache` into
`install_vllm_metal_paged_attention_kv_cache` would make `paged_gqa_sdpa_calls` non-zero and satisfy
criterion 5 *literally* while possibly moving criteria 3 and 4 not at all. That failure mode is
already measured once in this plan: the dense-decode ceiling flips the executed layout and changes
throughput by -0.30 % against a 1.61 % within-arm spread. A counter that fires is not a speedup.

**What the code says the cost can be.** `models/qwen4_exp.py:3760-3786`: the QSA layer builds a
selection mask from the indexer's pooled block keys, then `mx.take(k, sel_mask, axis=2)` /
`mx.take(v, ...)` **gathers the selected rows into a fresh array**, then runs
`mx.fast.scaled_dot_product_attention(q, k, v, mask=None)` on that subset. So the per-token work that
grows with context is the indexer selection over the pooled table plus the gather over a KV of size
ctx -- not a dense attention over the whole context. `QSACache` keeps `raw_keys`, `pooled` and
`pooled_f32_t` as positional buffers keyed to `kv.offset` (`:1950-2016`), grows them geometrically
after a documented O(N^2) incident (`_grown_cap`, `:1939`), and round-trips through the session bank
via its `state` property (`:2099-2107`).

- [x] **Step 1 — DONE 2026-09-06 23:30, and it found the gate rather than a knob list.** The QSA
      prefill pipeline is gated by `_qsa_prefill_enabled()` (`models/qwen4_exp.py:1480`), AUTO by
      default, resolving through `qsa_prefill_lane_auto_supported()` (`:1439`). Probing those in the
      serving venv returned `nax_available False`, `lane_auto_supported False`,
      `_qsa_prefill_enabled False`, with the engine's own diagnostic: `QSA sparse prefill disabled:
      this Metal SDK cannot compile the MPP score pipeline; using dense prefill`. The auto gate has
      exactly two fast consumers (`:1440-1453`): the Metal 4 TensorOps (NAX) flash kernel, which is
      M4/M5-only, and -- for M3-class GPUs, which have no G17 tensor units -- the vendored **Steel
      sparse-GQA** kernel "when it is built and probed". Neither was available, because:
      `_IMPORT_ERROR = ModuleNotFoundError("No module named 'mtplx_qsa_kernels'")`. The extension
      ships in this tree at `native_extensions/qsa_kernels` (oMLX PR #3244, Apache-2.0) with
      `CMakeLists.txt`, `setup.py`, `.metal` source and a README; its artifacts are gitignored, so a
      clone never carries them and no boot path warns that the fast lane is off. Knobs enumerated on
      the way, all read at gate time: `MTPLX_QSA_PREFILL` (master, auto),
      `MTPLX_QSA_PREFILL_MIN_ROWS` 32, `_MIN_CONTEXT` / `_FLASH_MIN_CONTEXT` / `_DIRECT_MIN_CONTEXT`
      32768, `MTPLX_QSA_GATHER` / `_GATHER_DECODE` / `_GATHER_MIN_CONTEXT` 16384 / `_GATHER_MAX_ROWS`
      8, `MTPLX_QSA_FLASH`, `MTPLX_QSA_SCORE_TILE_ROWS`, `MTPLX_FUSED_QSA_INDEXER`,
      `MTPLX_COMPILED_QSA_INDEXER`; model config `indexer_budget` 2048, `indexer_compress_ratio` 4.
- [x] **Step 2 — DONE 2026-09-06 23:45 by exactly the ablation this step prescribed.** One
      variable, `MTPLX_QSA_PREFILL` (auto versus `0`), verified present in the process environment
      with `ps eww`; matched arms on the side-by-side 9003, same pack / profile / depth / chunk,
      `--unique-body` so every cold row is genuinely cold, 9001 and 9002 untouched. At 104 k: cold
      251.33 s / 403.1 tok/s with the lane off versus **104.68 s median of 3 / 968.4 tok/s** with it
      on (-58.4 % wall, +140.6 % throughput); follow-up 2.79 s versus **1.56 s** median of 3; peak
      129.8 versus 114.9 GiB. At 52 k the same arms give +21.0 %, so the gain grows with context --
      the signature of removing a superlinear term, which is what the dense-mask reconstruction was.
      The cold marginal cost between 50.7 k and 101.3 k falls from 3.8499 to 2.0464 ms/token, and the
      follow-up still tracks it (ratio 1.138), so the prompt-breakdown conclusion survives the fix.
      No stopwatch inside the layer was needed and no synchronizing timer was added.
- [x] **Step 3 — resolved 2026-09-06 23:45 by a fourth option this step did not list.** None of
      (a) block-aligning the KV, (b) cheapening the selection, (c) holding context client-side was
      needed: the sparse selection already existed and was merely **not compiled**. The fix is a
      build, not a code change -- `uv pip install "nanobind==2.15.0" "setuptools>=42"` into the
      serving venv (the extension pins nanobind exactly; a mismatch imports cleanly and then rejects
      every `mx.array`, oMLX #2139), then `setup.py build_ext --inplace`, whose artifacts
      `qsa_prefill_direct.py:116-130` finds on its own. Proof: `BUILT_AGAINST_MLX 0.32.2` equals the
      imported mlx, `BUILT_AGAINST_NANOBIND 2.15.0`, `preflight pipeline ok: True`,
      `_qsa_prefill_enabled(): True`, `otool -L` resolving `@rpath/libmlx.dylib`, repo tree still
      clean. Two linker warnings recorded rather than hidden: a missing
      `/Applications/Xcode_26.6.app/...` framework search path, and "building for macOS-15.0, but
      linking with dylib `@rpath/libmlx.dylib` which was built for newer version 26.2". Options (a)
      and (b) stay unexplored and are now lower value; (c) is no longer the honest outcome for
      criterion 4. Recorded in this plan before any product code was written, as the step required.
- [ ] **Step 4 — PARTLY met 2026-09-06 23:45; open on criterion 3 and on durability.** Criterion 4
      is met: cold 103 k **104.68 s median of 3** against the 134 s target (267.920 s baseline).
      Criterion 3 is not: follow-up **1.56 s median of 3** against a 1.40 s target, 11.4 % above,
      though that is -44.4 % from the 2.805 s baseline. Still open here: (i) CLOSED 2026-09-07 00:49 -- both ports restarted one at a time and measured lane-on
      (cold 52 k: 9002 55.94 s / 905.9 tok/s against the 68.35 s lane-off reference, 9001
      57.02 s / 905.9 tok/s; follow-ups 1.45 s and 1.52 s with the breakdown complete); (ii) CLOSED 2026-09-07 01:05 -- `install_repo_launcher.sh` section 5 probes the lane through
      the serving venv, and `MISSING_EXT`/`MISMATCH` fail `--check` (exit 1) on any machine that
      can build the extension, print the recipe, and are reported-but-not-drift where no Metal
      toolchain exists; (iii) two pre-existing defects surfaced by the measurement
      and belong in their own slices: an identical repeat request 500s at `qwen4_fixed_verify.py:269`
      (`qwen4 fixed-M4 prompt history does not match the prefetched cache`, check introduced by
      `d6018d2`, ported from PR #391 -- and a repeat is a retry-shaped request), and a diverging
      continuation restores from a snapshot trimmed to 94 208 of 101 305 tokens and so re-prefills
      ~7.7 k tokens (36.57 s and 26.90 s measured, against 1.56 s for a continuation that extends
      the chain).

Known risks for (a), stated up front because they are the ones that will bite: the paged install's
contract is `keys`/`values` on the entry (`cache_state.py` install loop), which `QSACache` does not
expose; `VllmMetalPagedKVCache` owns a block table while `QSACache`'s indexer streams are positional
buffers keyed to `kv.offset`, so a conversion has to keep the two views consistent across
`trim(n)`, speculative rollback and the bank's `state` round-trip; and the session bank stores
snapshots of these caches, so any layout change has to survive a restore or warm prefixes will
silently diverge (the `Desktop QA, pre-v2` failure the restore code comments describe).

### Task 8 — an identical repeat request must not 500 (found while measuring, pre-existing)

Found 2026-09-06 23:30 while repeating a follow-up to collect a median of three: re-sending a
conversation whose whole prompt is already cached returns HTTP 500 with
`ValueError: qwen4 fixed-M4 prompt history does not match the prefetched cache`
(`qwen4_fixed_verify.py:269`, reached from `graphbank.py:2015 install_fixed_m4`). The check comes
from `d6018d2` (ported from PR #391), so it is not a regression from this work stream, and no test
covers it -- `grep 'prompt history does not match' tests/` is empty. A repeat is a retry-shaped
request, so a client can meet it in normal use.

Mechanism, read rather than guessed: the default aux route is `staged_sidecar`
(`MTPLX_COMPILED_VERIFY_BOUNDARY` defaults to `both`, `graphbank.py:1354-1358`), and its builder
compares the *device* history -- `previous = cache[ple_stage][ple.NGRAM_IDX]`, a `(1, 2)` int64
window (`qwen4_fixed_verify.py:209-216`) -- against the last two prompt tokens, because the host
ledger it stages must start from the same pair the cache holds. A finished generation leaves that
window advanced by the token it produced, and with a fully cached prompt no suffix forward runs to
re-align it. **The check itself is right**: staging a ledger from the prompt tail while the device
holds a different pair would be silently wrong. What is wrong is the consequence -- a 500.

The other route already solves it: `materialized` (`_prepare_fixed_m4_materialized` →
`_prepare_compiled_verify_aux`) reads `previous` from the cache slot itself, so it is consistent by
construction. And the choice is observable, not silent: `aux_route`/`aux_inputs` go into
`_fixed_m4_dispatch` and surface in the stats snapshot (`graphbank.py:2053`, `:3014-3019`).

- [x] **Step 1 — failing test first, and it failed for the right reason.**
      `test_fixed_m4_falls_back_to_materialized_when_device_history_is_ahead` in
      `tests/test_qwen4_exp_capture_commit.py`, on the existing two-route harness (`tm`, `_ids`,
      `_host_ids`, `_FakeSidecar`, `install_qwen4_fixed_verify_route`,
      `MTPLX_COMPILED_VERIFY_BOUNDARY=both`): it overwrites the PLE slot's n-gram window with a pair
      that is not the prompt tail, then asserts `install_fixed_m4` does not raise and reports
      `aux_route == "materialized"` / `aux_inputs == "device_history"`, while an untouched cache
      still reports `staged_sidecar` so the common path cannot regress quietly. Red first with the
      production traceback (`qwen4_fixed_verify.py:269: ValueError: qwen4 fixed-M4 prompt history
      does not match the prefetched cache`). It also pins the new counter at 1 for the fallback and
      0 for the aligned install.
- [x] **Step 2 — implemented.** The builder returns `None` for *this* mismatch instead of raising,
      and every geometry or sidecar raise stays loud: a wrong shape or a missing sidecar is a defect,
      not a route choice. `install_fixed_m4` reads `None` as "the staged route cannot start from this
      device history", takes the materialized branch, and counts it in
      `stats["fixed_m4_staged_history_fallbacks"]` so a fleet-wide move off the faster route is
      visible instead of inferred.
- [x] **Step 3 — live proof on the side-by-side 9003, 2026-09-07 01:34.** Request 1 cold: 200 in 8.43 s with
      `compiled_verify.fixed_m4.aux_route = staged_sidecar`, `aux_inputs = host_ledger`. Request 2,
      byte-identical: **200** in 0.10 s with `cached = 7863`, `new = 0`,
      `session_restore_mode = near_prefix_clone`, `aux_route = materialized`,
      `aux_inputs = device_history`. The route change is itself the evidence the mismatch was real --
      an aligned cache stays staged -- and request 1 shows the common path untouched. Before the
      change the same second request returned HTTP 500. 9001/9002 healthy throughout; 9003 stopped
      afterwards.
- [x] **Step 4 — full suite green at the commit carrying this** (`pytest tests/` exit 0, zero
      `FAILED`), and the receipt records before (500 + traceback) and after.

### QSA confirmation and boundary storage follow-up (2026-09-07)

Issue-1's "Conferma QSA e costo boundary" is the plan of record for this measurement slice.
Artifacts: `$MTPLX_HOME/bench/qsa-confirmation-20260907/`. Full results and corrected historical
attributions are in the receipt's "Corrected confirmation and boundary storage price" section.

- [x] Three real 104k baseline/compiled rounds, identical input hashes and explicit engagement:
      `confirmation-v2.jsonl`, exit 0. Compiled requires fused AND compiled flags; the earlier
      compiled-only screen did not establish engagement. Median follow-up: baseline 1.652785 s,
      candidate 1.882180 s. Compiled calls occur in cold/divergent legs, not the brief follow-up.
      **Criterion 3 remains open; do not enable this candidate for it.**
- [x] Short and long greedy comparisons agree across the three rounds of each arm. Evidence:
      `confirmation-v2.log` and stored response fingerprints. This is sampled output agreement,
      not universal state/logit parity.
- [x] Boundary storage measured on equivalent persisted cold entries: `boundary-single-cold.jsonl`,
      one entry per bank, matching prefix/token hash, 7 versus 28 retained boundaries. Raising cap
      8 to 32 adds 2.262108 GiB of logical snapshot tensors and 2.346069 GiB of allocated bank disk.
      Non-boundary tensor metadata/blob references match. RAM bank entry counts differed, so do
      not interpret aggregate RAM counters as marginal per-snapshot memory.
- [ ] Boundary output equivalence remains unproven: `boundary-memory.jsonl` has differing long
      greedy fingerprints at cap8/cap32. Cold-only storage equivalence does not close this finding.
      No production boundary-cap change or push is authorized by these measurements.

### Actual suffix-width follow-up (2026-09-07)

Issue-1 slice "Follow-up: compilazione della larghezza effettiva" tested the existing
`MTPLX_QSA_PREFILL_COMPILE_ROWS=256` setting with fused/compiled enabled, preserving boundary8
and tail grid256. Artifact root: `$MTPLX_HOME/bench/qsa-followup-256-20260907/`.

- [x] `indexer-tests.log`: 16 tests passed; actual 104k serving slice in `screen.jsonl`
      confirms 26 compiled selector calls in the ordinary follow-up and matching fingerprints.
- [x] Candidate assessed, not enabled: 1.441413 s against paired baseline1.385912 s supplies no
      positive speed signal. No claim of a statistically established regression from one pair.
- [x] Baseline repeated without another configuration change: `baseline-repeats.jsonl` plus the
      baseline in `screen.jsonl` give three samples with median1.385911626 s, identical work and
      outputs. Numeric threshold met in this ~101k subcase, not a new software improvement.
- [ ] Original criterion3 remains broader: original103k/556-token work and disappearance of
      context scaling have not been demonstrated by the current sample. Do not mark it fully met
      or attribute cross-session baseline variation to an unverified cause.

### Consolidation decision (2026-09-07)

The user elected to stop further tuning at diminishing returns and commit the measured result.
Keep the already-enabled sparse QSA lane; do not enable the experimental compiled-indexer or
boundary32 arms. The remaining acceptance gaps above are deferred, not marked satisfied. No
additional server restart, production configuration change, or push is part of this consolidation.

