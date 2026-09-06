#!/usr/bin/env python3
"""Repeat the follow-up request N times against an already-warm session bank.

Criterion 3 of docs/plans/2026-09-06-long-context-prefill-handoff.md wants a
median of >=3 runs, and `prefill_probe.py` sends one follow-up per rung. This
reuses the probe's own helpers so the prefix text is byte-identical to the rung
that filled the bank, and prints the engine rows with the prompt-phase
breakdown (`prompt_suffix_time_s` and friends).

Each repeat varies the follow-up text (`-fu2`, `-fu3`, ...). An *identical*
repeat is not usable: with the whole prompt already cached there are zero new
tokens and the request 500s in
`qwen4_fixed_verify._build_fixed_m4_compiled_verify_aux` with "qwen4 fixed-M4
prompt history does not match the prefetched cache" -- the n-gram sidecar is left
advanced by the previous generation and nothing re-aligns it when there is no
suffix. Pre-existing (the check comes from commit d6018d2, ported from PR #391).
Varying the text measures the same work -- a ~675-token suffix at the same
retained context -- without that path.

  python3 scripts/followup_repeat.py --port 9003 --tag qsaA --rung 106496 --repeats 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import prefill_probe as probe

MTPLX_HOME = Path(os.environ.get("MTPLX_HOME", Path.home() / ".mtplx"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True, help="the probe tag that filled the bank")
    ap.add_argument("--rung", type=int, required=True, help="target tokens of that rung")
    ap.add_argument("--repeats", type=int, default=2, help="extra follow-ups to send")
    ap.add_argument("--followup-tokens", type=int, default=700)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    log = MTPLX_HOME / "logs" / f"request-log-{args.port}.jsonl"

    seed = f"{probe.SALT}-{args.tag}-{args.rung}"
    body = probe.unique_filler(args.rung, seed)
    msgs = [{"role": "user", "content": f"[probe {seed}]\n{body}"}]

    offset = log.stat().st_size if log.exists() else 0
    for i in range(args.repeats):
        payload = {
            "messages": msgs + [
                {
                    "role": "user",
                    "content": probe.unique_filler(
                        args.followup_tokens, seed + f"-fu{i + 2}"
                    ),
                }
            ],
            "max_tokens": 1,
            "stream": False,
        }
        wall = probe.request(base, payload, 900.0)
        print(f"follow-up #{i + 2} wall={wall:6.2f}s", flush=True)

    with log.open() as handle:
        handle.seek(offset)
        rows = [json.loads(line) for line in handle if line.strip()]

    for r in rows:
        new = r.get("new_prefill_tokens") or 0
        ev = r.get("prompt_eval_time_s") or 0.0
        print(
            f"  new={new} cached={r.get('cached_tokens')} "
            f"eval={round(ev, 3)}s "
            f"suffix={round(r.get('prompt_suffix_time_s') or 0.0, 3)}s "
            f"repair={round(r.get('prompt_repair_time_s') or 0.0, 4)}s "
            f"tok/s={round(new / ev, 1) if ev else 0} "
            f"complete={r.get('prompt_eval_breakdown_complete')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
