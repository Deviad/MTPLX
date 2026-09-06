#!/usr/bin/env python3
"""Screen the opt-in QSA kernels against the remaining criterion-3 gap.

The QSA sparse prefill lane is on by default once its native extension is built,
and it took the 103 k follow-up from 2.805 s to 1.56 s -- still 11.4 % above the
1.40 s target that `docs/plans/2026-09-06-long-context-prefill-handoff.md` sets.
Four opt-in kernels around the indexer and the gather are off by default and no
profile sets them, so they are the levers that are left. This screens them one
env at a time on a side-by-side port and checks output exactness beside speed,
because three of them are declared *exact* selectors: a kernel that is faster and
different is not a candidate, it is a regression with good manners.

Discipline, the same as the lane sweep this reuses: one arm per server process,
the session bank emptied per arm so every cold row is genuinely cold, and the
arm's own env recorded in the row so the table cannot be read without it.

  python3 scripts/qsa_optin_battery.py --port 9003 --contexts 104k
  python3 scripts/qsa_optin_battery.py --port 9003 --contexts 104k --rounds 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prefill_lane_sweep as sweep  # noqa: E402  (stop/boot/row helpers)
import prefill_probe as probe  # noqa: E402  (filler text + SALT)

MTPLX_HOME = Path(os.environ.get("MTPLX_HOME", Path.home() / ".mtplx"))

# Off by default in the model, registered as overridable in profiles.py, and not
# set by any profile -- which is exactly why they are still unmeasured here.
ARMS: dict[str, dict[str, str]] = {
    "baseline": {},
    "gather": {"MTPLX_QSA_GATHER": "1"},
    "fused_indexer": {"MTPLX_FUSED_QSA_INDEXER": "1"},
    "compiled_indexer": {"MTPLX_COMPILED_QSA_INDEXER": "1"},
}

# Fixed prompt for the exactness leg: short, so it costs seconds, and greedy, so
# a differing completion is a real divergence rather than sampler noise.
EXACTNESS_PROMPT = (
    "Reply with exactly one sentence naming the three primary colours, "
    "in alphabetical order, separated by commas."
)
EXACTNESS_TOKENS = 64

ROW_FIELDS = (
    "prompt_tokens",
    "cached_tokens",
    "new_prefill_tokens",
    "prompt_eval_time_s",
    "prompt_suffix_time_s",
    "prompt_repair_time_s",
    "prompt_mtp_history_time_s",
    "prompt_eval_breakdown_complete",
    "prefill_layout",
    "prefill_attention_impl",
    "peak_memory_bytes",
    "session_restore_mode",
)


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode())


def completion_fingerprint(response: dict) -> str:
    """Everything a greedy run can differ in, as one comparable string.

    `content` alone is not enough, and the first rehearsal showed why: with
    reasoning enabled the model can spend the whole token budget thinking and
    return an empty `content`, so every arm "matched" on an empty string. The
    fingerprint carries the reasoning text and the finish reason beside it.
    """
    try:
        choice = response["choices"][0]
    except (KeyError, IndexError, TypeError):
        return json.dumps(response, sort_keys=True)[:800]
    message = choice.get("message") or {}
    return json.dumps(
        {
            "finish_reason": choice.get("finish_reason"),
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
            "tool_calls": message.get("tool_calls"),
        },
        sort_keys=True,
    )


def rows_after(offset: int) -> list[dict]:
    out = []
    if not sweep.LOG_ROWS.exists():
        return out
    with sweep.LOG_ROWS.open() as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def run_arm(arm: str, rnd: int, contexts: str, boot_limit_s: int) -> dict:
    env_overrides = ARMS[arm]
    sweep.stop_server()
    if sweep.BANK.exists():
        shutil.rmtree(sweep.BANK, ignore_errors=True)
    sweep.BANK.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(env_overrides)
    serve_log = MTPLX_HOME / "logs" / f"qsa-battery-{arm}-r{rnd}-serve.log"
    handle = serve_log.open("w")
    proc = subprocess.Popen(
        ["/bin/bash", str(sweep.SERVE)],
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rec: dict = {
        "arm": arm,
        "round": rnd,
        "pid": proc.pid,
        "arm_env": dict(env_overrides),
        "serve_log": str(serve_log),
    }
    try:
        if not sweep.wait_healthy(boot_limit_s):
            rec["error"] = "server did not become healthy"
            return rec
        base = f"http://127.0.0.1:{sweep.PORT}"

        # Speed leg: one cold prefill plus the follow-up that criterion 3 is about.
        offset = sweep.LOG_ROWS.stat().st_size if sweep.LOG_ROWS.exists() else 0
        tag = f"qsabatt-{arm}"
        probe_out = sweep.sh(
            [
                sys.executable, str(sweep.PROBE), "--port", str(sweep.PORT),
                "--contexts", contexts, "--tag", tag, "--unique-body",
                "--max-new-tokens", "1",
            ],
            timeout=2400,
        )
        rec["probe_tail"] = probe_out.strip().splitlines()[-1] if probe_out.strip() else ""
        rows = rows_after(offset)
        cold = [r for r in rows if (r.get("cached_tokens") or 0) == 0]
        warm = [r for r in rows if (r.get("cached_tokens") or 0) > 0]
        for label, picked in (("cold", cold[-1] if cold else {}),
                              ("followup", warm[-1] if warm else {})):
            for key in ROW_FIELDS:
                rec[f"{label}_{key}"] = picked.get(key)
            new = picked.get("new_prefill_tokens") or 0
            ev = picked.get("prompt_eval_time_s") or 0
            rec[f"{label}_ms_per_new_token"] = round(ev * 1000 / new, 3) if new and ev else None

        # Exactness leg: greedy, fixed prompt, compared across arms afterwards.
        offset2 = sweep.LOG_ROWS.stat().st_size if sweep.LOG_ROWS.exists() else 0
        response = post_json(
            f"{base}/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": EXACTNESS_PROMPT}],
                "max_tokens": EXACTNESS_TOKENS,
                "temperature": 0,
                "stream": False,
            },
            600.0,
        )
        rec["exactness_fingerprint"] = completion_fingerprint(response)
        rec["exactness_bytes"] = len(rec["exactness_fingerprint"])
        exact_rows = rows_after(offset2)
        if exact_rows:
            fm4 = (exact_rows[-1].get("compiled_verify") or {}).get("fixed_m4") or {}
            rec["exactness_aux_route"] = fm4.get("aux_route")
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001 - one arm failing must not kill the run
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sweep.stop_server()
        handle.close()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=9003)
    ap.add_argument("--contexts", default="104k")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--boot-limit-s", type=int, default=420)
    ap.add_argument("--arms", default="", help="comma-separated subset of arm names")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sweep.configure(
        port=args.port,
        serve=os.environ.get("MTPLX_LANE_SWEEP_SERVE", ""),
        bank=os.environ.get("MTPLX_LANE_SWEEP_BANK", ""),
        probe=os.environ.get("MTPLX_LANE_SWEEP_PROBE", ""),
        contexts=args.contexts,
    )
    names = [a.strip() for a in args.arms.split(",") if a.strip()] or list(ARMS)
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        ap.error(f"unknown arms: {unknown}; known: {sorted(ARMS)}")

    out = Path(args.out or (MTPLX_HOME / "bench" /
                            f"qsa-battery-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"qsa opt-in battery -> {out}", flush=True)
    print(f"arms: {names} | contexts {args.contexts} | rounds {args.rounds}", flush=True)

    records = []
    for rnd in range(1, args.rounds + 1):
        for arm in names:
            print(f"[round {rnd}] {arm}: booting ...", flush=True)
            rec = run_arm(arm, rnd, args.contexts, args.boot_limit_s)
            records.append(rec)
            with out.open("a") as handle:
                handle.write(json.dumps(rec, sort_keys=True) + "\n")
            if rec.get("error"):
                print(f"[round {rnd}] {arm}: ERROR {rec['error']}", flush=True)
            else:
                print(
                    f"[round {rnd}] {arm}: cold={rec.get('cold_prompt_eval_time_s')}s "
                    f"followup={rec.get('followup_prompt_eval_time_s')}s "
                    f"ms/new={rec.get('followup_ms_per_new_token')}",
                    flush=True,
                )

    print()
    header = (f"{'arm':<18} {'cold_s':>8} {'fu_s':>7} {'fu ms/new':>10} "
              f"{'cold tok/s':>11} {'exactness':>10}")
    print(header)
    reference = next((r.get("exactness_fingerprint") for r in records
                      if r.get("arm") == "baseline" and r.get("exactness_fingerprint")), None)
    for rec in records:
        cold_new = rec.get("cold_new_prefill_tokens") or 0
        cold_ev = rec.get("cold_prompt_eval_time_s") or 0
        cold_rate = round(cold_new / cold_ev, 1) if cold_new and cold_ev else None
        text = rec.get("exactness_fingerprint")
        if not text:
            same = "no answer"
        elif reference is None:
            same = "no baseline"
        else:
            same = "match" if text == reference else "DIFFERS"
        print(f"{rec.get('arm','?'):<18} "
              f"{str(round(cold_ev, 2) if cold_ev else None):>8} "
              f"{str(round(rec.get('followup_prompt_eval_time_s') or 0, 3) or None):>7} "
              f"{str(rec.get('followup_ms_per_new_token')):>10} "
              f"{str(cold_rate):>11} {same:>10}")
    if reference is None:
        print("\nno baseline fingerprint: the greedy comparison could not be made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
