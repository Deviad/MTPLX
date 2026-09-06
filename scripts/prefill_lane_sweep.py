#!/usr/bin/env python3
"""Lane sweep: does the dense-decode ceiling change the layout qwen4_exp executes?

Three arms against a fresh repo-build server on 9003, interleaved over rounds so
server-start drift hits every arm:

  auto     nothing set -- whatever the profile picks at 62.9k
  ceiling  MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=32768 (the knob the 16:36
           sweep measured a +0.40% non-effect for, without being able to say
           which layout ran)
  repage   MTPLX_SUSTAINED_PREFILL_LAYOUT=contiguous_then_repage -- the positive
           control. If `prefill_layout` never moves, the instrument is dead and
           "none everywhere" means nothing; if it moves here, the field reads
           the executed path.

Every arm uses the SAME probe tag, so the prompt body is byte-identical (the
probe salts its filler with the tag), and the session bank is emptied per arm
so every row is a cold prefill.

  python3 scripts/prefill_lane_sweep.py --rounds 3 --port 9003 --contexts 64k
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

MTPLX_HOME = Path(os.environ.get("MTPLX_HOME", Path.home() / ".mtplx"))
HERE = Path(__file__).resolve().parent

# The defaults reproduce the side-by-side run these were written for: a
# repo-build server on 9003 with its own session bank. The serve wrapper is
# operator config (which pack, which port, which profile) and is deliberately
# NOT published with this script, so --serve is how a run names it.
PORT = 9003
LOG_ROWS = MTPLX_HOME / "logs" / f"request-log-{PORT}.jsonl"
BANK = MTPLX_HOME / "session-bank" / f"flash-next-repo-{PORT}"
SERVE = MTPLX_HOME / "scripts" / f"serve-repo-{PORT}.sh"
PROBE = HERE / "prefill_probe.py"
RUN_TAG = "lanematch"          # fixed: identical prompt bytes across arms
CONTEXTS = "64k"


def configure(*, port: int, serve: str = "", bank: str = "", probe: str = "",
              contexts: str = "") -> None:
    """Rebind the run's targets once, before any arm boots.

    Empty values resolve to the documented defaults, so this is also the entry
    point for other tools that drive the same server (the opt-in kernel battery
    imports it rather than re-implementing stop/boot/row-reading). One place
    composes the defaults: a caller cannot pass a half-configured run.
    """
    global PORT, LOG_ROWS, BANK, SERVE, PROBE, CONTEXTS
    PORT = int(port)
    LOG_ROWS = MTPLX_HOME / "logs" / f"request-log-{PORT}.jsonl"
    BANK = Path(bank) if bank else MTPLX_HOME / "session-bank" / f"flash-next-repo-{PORT}"
    SERVE = Path(serve) if serve else MTPLX_HOME / "scripts" / f"serve-repo-{PORT}.sh"
    PROBE = Path(probe) if probe else HERE / "prefill_probe.py"
    CONTEXTS = contexts or CONTEXTS

ARMS: dict[str, dict[str, str]] = {
    "auto": {},
    "ceiling": {"MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT": "32768"},
    "repage": {"MTPLX_SUSTAINED_PREFILL_LAYOUT": "contiguous_then_repage"},
}

FIELDS = (
    "prompt_tokens",
    "new_prefill_tokens",
    "cached_tokens",
    "prompt_eval_time_s",
    "prefill_tok_s",
    "prefill_attention_impl",
    "prefill_layout",
    "prefill_route",
    "prefill_partitioned_paged_calls",
    "prefill_dense_fallback_calls",
    "prefill_large_q_split_sdpa_fallback_calls",
    "paged_gqa_sdpa_calls",
    "peak_memory_bytes",
)


def sh(args: list[str], env: dict | None = None, timeout: int = 120) -> str:
    return subprocess.run(
        args, capture_output=True, text=True, env=env, timeout=timeout
    ).stdout


def listener_pid() -> str:
    return sh(["lsof", "-nP", "-iTCP:%d" % PORT, "-sTCP:LISTEN", "-t"]).strip()


def stop_server() -> None:
    pid = listener_pid()
    if not pid:
        return
    subprocess.run(
        [str(MTPLX_HOME / "bin" / "mtplx"), "stop", "--port", str(PORT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    for _ in range(40):
        if not listener_pid():
            return
        time.sleep(1.5)
    pid = listener_pid()
    if pid:
        os.kill(int(pid), signal.SIGKILL)
        time.sleep(2)


def wait_healthy(limit_s: int) -> bool:
    deadline = time.time() + limit_s
    while time.time() < deadline:
        code = sh(
            ["curl", "-s", "-m", "4", "-o", "/dev/null", "-w", "%{http_code}",
             f"http://127.0.0.1:{PORT}/health"]
        ).strip()
        if code == "200":
            return True
        time.sleep(3)
    return False


def cold_row(offset: int) -> dict:
    """The newest row after `offset` whose cached_tokens is 0 (the cold prefill)."""
    rows = []
    with LOG_ROWS.open() as handle:
        handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    cold = [r for r in rows if (r.get("cached_tokens") or 0) == 0]
    return cold[-1] if cold else {}


def run_arm(arm: str, rnd: int, boot_limit_s: int) -> dict:
    stop_server()
    if BANK.exists():
        shutil.rmtree(BANK, ignore_errors=True)
    BANK.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(ARMS[arm])
    serve_log = MTPLX_HOME / "logs" / f"lane-sweep-{arm}-r{rnd}-serve.log"
    handle = serve_log.open("w")
    proc = subprocess.Popen(
        ["/bin/bash", str(SERVE)], env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rec: dict = {"arm": arm, "round": rnd, "pid": proc.pid, "serve_log": str(serve_log)}
    try:
        if not wait_healthy(boot_limit_s):
            rec["error"] = "server did not become healthy"
            return rec
        offset = LOG_ROWS.stat().st_size if LOG_ROWS.exists() else 0
        probe = sh(
            [sys.executable, str(PROBE), "--port", str(PORT), "--contexts", CONTEXTS,
             "--tag", RUN_TAG, "--max-new-tokens", "1"],
            timeout=2400,
        )
        rec["probe_tail"] = probe.strip().splitlines()[-1] if probe.strip() else ""
        row = cold_row(offset)
        if not row:
            rec["error"] = "no cold engine row"
            return rec
        for key in FIELDS:
            rec[key] = row.get(key)
        new, ev = rec.get("new_prefill_tokens") or 0, rec.get("prompt_eval_time_s") or 0
        rec["implied_new_tok_s"] = round(new / ev, 1) if new and ev else None
        # The arm's own claim about which layout it asked for, straight from env.
        rec["arm_env"] = dict(ARMS[arm])
    finally:
        stop_server()
        handle.close()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--boot-limit-s", type=int, default=420)
    ap.add_argument("--out", default="")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--contexts", default=CONTEXTS)
    ap.add_argument("--serve", default="",
                    help="serve wrapper booted per arm "
                         "(default: $MTPLX_HOME/scripts/serve-repo-<port>.sh)")
    ap.add_argument("--bank", default="",
                    help="session bank emptied per arm "
                         "(default: $MTPLX_HOME/session-bank/flash-next-repo-<port>)")
    ap.add_argument("--probe", default="",
                    help="prefill_probe.py to drive (default: this script's sibling)")
    args = ap.parse_args()

    configure(
        port=args.port,
        serve=args.serve,
        bank=args.bank,
        probe=args.probe,
        contexts=args.contexts,
    )
    if not SERVE.exists():
        ap.error(f"--serve not found: {SERVE} (the serve wrapper is operator config)")
    if not PROBE.exists():
        ap.error(f"--probe not found: {PROBE}")

    names = list(ARMS)
    out = Path(args.out or (MTPLX_HOME / "bench" /
                            f"lane-sweep-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"lane sweep -> {out}", flush=True)

    records = []
    for rnd in range(1, args.rounds + 1):
        order = names[rnd % len(names):] + names[: rnd % len(names)]  # rotate, not repeat
        for arm in order:
            print(f"[round {rnd}] {arm}: booting ...", flush=True)
            rec = run_arm(arm, rnd, args.boot_limit_s)
            records.append(rec)
            with out.open("a") as handle:
                handle.write(json.dumps(rec, sort_keys=True) + "\n")
            if rec.get("error"):
                print(f"[round {rnd}] {arm}: ERROR {rec['error']}", flush=True)
            else:
                print(
                    f"[round {rnd}] {arm}: implied={rec['implied_new_tok_s']} tok/s "
                    f"impl={rec.get('prefill_attention_impl')!r} "
                    f"layout={rec.get('prefill_layout')!r}",
                    flush=True,
                )

    print("\n=== riassunto ===", flush=True)
    for arm in names:
        rows = [r for r in records if r["arm"] == arm and not r.get("error")]
        vals = [r["implied_new_tok_s"] for r in rows if r.get("implied_new_tok_s")]
        impls = sorted({str(r.get("prefill_attention_impl")) for r in rows})
        layouts = sorted({str(r.get("prefill_layout")) for r in rows})
        mean = sum(vals) / len(vals) if vals else 0.0
        spread = (max(vals) - min(vals)) / mean * 100 if vals and mean else 0.0
        print(
            f"{arm:8} n={len(rows)} implied={mean:7.1f} tok/s spread={spread:4.2f}%  "
            f"impl={impls} layout={layouts}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
