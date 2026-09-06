#!/usr/bin/env python3
"""Does a follow-up's cost depend on the retained context or on the suffix width?

Sends follow-ups of different requested widths against the same cached prefix
and prints the per-new-token cost from the engine rows. If the cost fell as the
suffix widened, the long-context follow-up would be a narrow-q
arithmetic-intensity problem; measured on the Flash-Next pack at ~101 k it is
flat (4.448-4.853 ms per new token across real suffixes of 7 220-9 006), so the
cost is set by the context the request sits at.

Read the `new=` column, not the requested width: the bank restores from a stored
snapshot that may be **trimmed** relative to the live prefix, so a continuation
that diverges from the stored chain re-prefills the difference. Measured on a
101 k prompt: the live prefix matched 101 305 tokens, the stored snapshot
restored at 94 208, and the real suffix became ~7.7 k tokens instead of the
requested 64-4 096. That is a finding in its own right (a diverging continuation
cost 36.57 s where an extending one cost 1.56 s), and it is why this script
prints what the engine actually evaluated.

  python3 scripts/prefill_probe.py --port 9003 --contexts 104k --tag sweep --unique-body
  python3 scripts/suffix_width_sweep.py --port 9003 --tag sweep --rung 106496
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
    ap.add_argument("--widths", default="64,256,1024,4096", help="requested suffix tokens")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    log = MTPLX_HOME / "logs" / f"request-log-{args.port}.jsonl"
    widths = [int(w) for w in args.widths.split(",") if w.strip()]

    seed = f"{probe.SALT}-{args.tag}-{args.rung}"
    body = probe.unique_filler(args.rung, seed)
    msgs = [{"role": "user", "content": f"[probe {seed}]\n{body}"}]

    offset = log.stat().st_size if log.exists() else 0
    for width in widths:
        payload = {
            "messages": msgs + [
                {
                    "role": "user",
                    "content": probe.unique_filler(width, seed + f"-w{width}"),
                }
            ],
            "max_tokens": 1,
            "stream": False,
        }
        wall = probe.request(base, payload, 900.0)
        print(f"larghezza richiesta {width:>5} -> wall {wall:7.2f}s", flush=True)

    with log.open() as handle:
        handle.seek(offset)
        rows = [json.loads(line) for line in handle if line.strip()]

    print()
    print(
        f"{'new':>6} {'cached':>8} {'eval_s':>8} {'suffix_s':>9} "
        f"{'ms/nuovo':>9} {'tok/s':>8} {'completo':>9}"
    )
    for r in rows:
        new = r.get("new_prefill_tokens") or 0
        if not new:
            continue
        ev = r.get("prompt_eval_time_s") or 0.0
        su = r.get("prompt_suffix_time_s") or 0.0
        print(
            f"{new:>6} {r.get('cached_tokens'):>8} {ev:>8.3f} {su:>9.3f} "
            f"{su * 1000 / new:>9.2f} {new / su if su else 0:>8.1f} "
            f"{str(r.get('prompt_eval_breakdown_complete')):>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
