#!/usr/bin/env python3
"""Compare QSA routes and boundary budgets on an isolated serving port.

The operator wrapper must honor MTPLX_BENCH_BANK. Banks and raw receipts are
retained beside --out. Only the process launched by this script is stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request

import prefill_lane_sweep as sweep
import prefill_probe as probe

ARMS = {
    "baseline": {},
    "gather": {"MTPLX_QSA_GATHER": "1"},
    "fused_indexer": {"MTPLX_FUSED_QSA_INDEXER": "1"},
    "compiled_indexer": {
        "MTPLX_FUSED_QSA_INDEXER": "1",
        "MTPLX_COMPILED_QSA_INDEXER": "1",
    },
    "compiled_suffix256": {
        "MTPLX_FUSED_QSA_INDEXER": "1",
        "MTPLX_COMPILED_QSA_INDEXER": "1",
        "MTPLX_QSA_PREFILL_COMPILE_ROWS": "256",
    },
    "boundary32": {"MTPLX_GDN_BOUNDARY_MAX": "32"},
}
CONTROL_ENV = {
    "MTPLX_QSA_GATHER": "0",
    "MTPLX_FUSED_QSA_INDEXER": "0",
    "MTPLX_COMPILED_QSA_INDEXER": "0",
    "MTPLX_QSA_PREFILL_COMPILE_ROWS": "2048",
    "MTPLX_QSA_SCORE_TILE_ROWS": "0",
    "MTPLX_QSA_PREFILL": "1",
    "MTPLX_GDN_BOUNDARY_MAX": "8",
    "MTPLX_GDN_BOUNDARY_TAIL_INTERVAL": "256",
    "MTPLX_SHUTDOWN_SSD_FLUSH_S": "60",
}
RUN_TAG = "qsa-battery"
SHORT_QUESTION = "Name the three primary colours in alphabetical order."
LONG_QUESTION = (
    "Considering everything above, state which single word appears most often "
    "in it, then repeat that word in parentheses."
)


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def health() -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{sweep.PORT}/health", timeout=15) as response:
        return json.load(response)


def completion_fingerprint(response: dict) -> str:
    choice = response["choices"][0]
    message = choice["message"]
    if not any(message.get(k) for k in ("content", "reasoning_content", "tool_calls")):
        raise ValueError("empty completion: exactness cannot be compared")
    return json.dumps({
        "finish_reason": choice["finish_reason"],
        **{k: message.get(k) for k in ("content", "reasoning_content", "tool_calls")},
    }, sort_keys=True)


def arm_environment(arm: str, bank: Path, engagement: Path) -> dict[str, str]:
    return {
        **os.environ, **CONTROL_ENV, **ARMS[arm],
        "MTPLX_BENCH_BANK": str(bank),
        "MTPLX_QSA_PREFILL_ENGAGEMENT_FILE": str(engagement),
    }


def dir_bytes(path: Path) -> int:
    result = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=True)
    return int(result.stdout.split()[0]) * 1024


def bank_state() -> dict:
    return health()["session_bank"]


def wait_for_writer_queue(limit_s: int = 300) -> dict:
    """Drain staged writes; scheduler-held persistence is flushed at shutdown."""
    deadline = time.monotonic() + limit_s
    while True:
        state = bank_state()
        cold = state["cold_tier"]
        if not cold["writer_queue_depth"] and not cold["writer_backlog_bytes"]:
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("SSD writer has not drained; bank bytes are not settled")
        time.sleep(2)


def engagement_counts(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def request_leg(messages: list[dict], max_tokens: int, dest: Path, engagement: Path) -> dict:
    offset = sweep.LOG_ROWS.stat().st_size if sweep.LOG_ROWS.exists() else 0
    counts_before = engagement_counts(engagement)
    started = time.monotonic()
    response = post_json(f"http://127.0.0.1:{sweep.PORT}/v1/chat/completions", {
        "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False,
    })
    elapsed = time.monotonic() - started
    dest.write_text(json.dumps(response, indent=2) + "\n")
    rows = []
    for _ in range(50):
        with sweep.LOG_ROWS.open() as handle:
            handle.seek(offset)
            appended = [json.loads(line) for line in handle if line.strip()]
            rows = [row for row in appended if row.get("request_id") == response["id"]]
            if any(row.get("request_id") not in (None, response["id"]) for row in appended):
                raise ValueError("another client used the benchmark port")
        if rows:
            break
        time.sleep(0.1)
    if len(rows) != 1:
        raise ValueError(f"expected one exclusive request row, got {len(rows)}")
    counts_after = engagement_counts(engagement)
    return {
        "wall_s": elapsed, "row": rows[0],
        "response_path": str(dest),
        "prompt_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        "fingerprint": completion_fingerprint(response),
        "engagement": {k: v - counts_before.get(k, 0) for k, v in counts_after.items()},
    }


def run_arm(arm: str, rnd: int, tokens: int, output_dir: Path, boot_limit_s: int,
            memory_only: bool = False) -> dict:
    if sweep.listener_pid():
        raise RuntimeError(f"port {sweep.PORT} already occupied; refusing to stop its owner")
    work = Path(tempfile.mkdtemp(prefix=f"{arm}-r{rnd}-", dir=output_dir))
    bank = work / "bank"
    bank.mkdir()
    engagement = work / "engagement.json"
    env = arm_environment(arm, bank, engagement)
    record = {
        "arm": arm, "round": rnd, "work_dir": str(work),
        "started_at": time.time(),
        "swap_before": subprocess.run(["sysctl", "-n", "vm.swapusage"],
                                      capture_output=True, text=True, check=True).stdout.strip(),
        "arm_env": {k: env[k] for k in CONTROL_ENV}, "ok": False,
    }
    seed = f"{probe.SALT}-{RUN_TAG}-{tokens}"
    messages = [{"role": "user", "content": f"[probe {seed}]\n{probe.unique_filler(tokens, seed)}"}]
    legs = [
        ("cold", messages, 1),
        ("followup", messages + [{"role": "user", "content": probe.unique_filler(700, seed + "-fu")}], 1),
        ("short", [{"role": "user", "content": SHORT_QUESTION}], 64),
        ("long", messages + [{"role": "user", "content": LONG_QUESTION}], 48),
    ]
    if memory_only:
        legs = legs[:1]
    with (work / "serve.log").open("w") as log:
        proc = subprocess.Popen(["/bin/bash", str(sweep.SERVE)], env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = proc.pid
        try:
            if not sweep.wait_healthy(boot_limit_s):
                raise TimeoutError("server did not become healthy")
            startup = health()
            (work / "startup-health.json").write_text(json.dumps(startup, indent=2) + "\n")
            if Path(startup["session_bank"]["cold_tier"]["dir"]).resolve() != bank.resolve():
                raise ValueError("wrapper did not honor MTPLX_BENCH_BANK; no requests sent")
            for name, payload, count in legs:
                leg = request_leg(payload, count, work / f"{name}-response.json", engagement)
                record[name] = leg
                if name == "cold" and leg["row"]["cached_tokens"] != 0:
                    raise ValueError("cold leg hit a cache")
                if name in ("followup", "long") and leg["row"]["cached_tokens"] <= 0:
                    raise ValueError(f"{name} leg missed the long prefix")
                print(f"  {arm} r{rnd} {name}: eval={leg['row']['prompt_eval_time_s']:.3f}s "
                      f"cached={leg['row']['cached_tokens']} new={leg['row']['new_prefill_tokens']} "
                      f"compiled={leg['engagement'].get('compiled_selector', 0)}", flush=True)
            record["bank_state"] = wait_for_writer_queue()
            record["bank_bytes_after"] = dir_bytes(bank)
            record["engagement"] = engagement_counts(engagement)
            # Small rungs only rehearse plumbing; they cannot establish engagement at depth.
            if arm == "compiled_indexer" and tokens >= 32768:
                if record["engagement"].get("compiled_selector", 0) <= 0:
                    raise ValueError("compiled arm did not execute the compiled selector")
            if arm == "compiled_suffix256" and tokens >= 32768 and not memory_only:
                if record["followup"]["engagement"].get("compiled_selector", 0) <= 0:
                    raise ValueError("suffix256 arm did not execute compiled selection in the follow-up")
            record["ok"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                record["shutdown_forced"] = True
            record["server_exit"] = proc.returncode
            record["bank_bytes_stopped"] = dir_bytes(bank)
            record["finished_at"] = time.time()
    if memory_only and record["ok"]:
        with sqlite3.connect(f"file:{bank / 'manifest.sqlite'}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            record["persisted_entries"] = [dict(row) for row in db.execute(
                "SELECT prefix_len, token_hash, entry_dir, nbytes, logical_nbytes, "
                "physical_nbytes FROM entries"
            )]
        if len(record["persisted_entries"]) != 1:
            record["ok"] = False
            record["error"] = "single-cold memory slice must persist exactly one entry"
    (work / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def comparisons(records: list[dict]) -> bool:
    baselines = [r for r in records if r["arm"] == "baseline" and r["ok"]]
    if not baselines:
        print("No successful baseline: comparison unavailable")
        return False
    reference = baselines[0]
    valid = True
    for record in records:
        if not record["ok"]:
            print(record["arm"], record.get("error"))
            valid = False
            continue
        if "persisted_entries" in reference:
            ref_entry = reference["persisted_entries"][0]
            entries = record.get("persisted_entries", [])
            same_entry = len(entries) == 1 and all(
                entries[0][key] == ref_entry[key] for key in ("prefix_len", "token_hash")
            )
            print(f"{record['arm']}: same_persisted_tokens={same_entry}")
            valid = valid and same_entry
        for leg in ("cold", "followup", "short", "long"):
            if leg not in reference and leg not in record:
                continue
            same_input = record[leg]["prompt_sha256"] == reference[leg]["prompt_sha256"]
            same_output = record[leg]["fingerprint"] == reference[leg]["fingerprint"]
            print(f"{record['arm']} r{record['round']} {leg}: "
                  f"same_input={same_input} same_output={same_output}")
            valid = valid and same_input and same_output
    return valid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=9003)
    ap.add_argument("--contexts", default="104k")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--boot-limit-s", type=int, default=420)
    ap.add_argument("--arms", default="baseline,compiled_indexer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--memory-only", action="store_true",
                    help="One cold request per arm, then verify persisted entry identity")
    args = ap.parse_args()
    names = args.arms.split(",")
    rungs = probe.parse_rungs(args.contexts)
    if len(rungs) != 1 or args.rounds < 1 or any(n not in ARMS for n in names):
        ap.error("use one rung, positive rounds, and known arms")
    if args.port in (9001, 9002):
        ap.error("production ports are excluded from this battery")
    sweep.configure(port=args.port, serve=os.environ.get("MTPLX_LANE_SWEEP_SERVE", ""))
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out.open("x") as handle:
        for rnd in range(1, args.rounds + 1):
            # Reverse alternate rounds to avoid always putting the candidate last.
            for arm in names if rnd % 2 else reversed(names):
                print(f"[round {rnd}] {arm}: booting", flush=True)
                record = run_arm(arm, rnd, rungs[0], out.parent, args.boot_limit_s,
                                 memory_only=args.memory_only)
                records.append(record)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                if not record["ok"]:
                    print(record.get("error"), flush=True)
                    return 1
    return 0 if comparisons(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
