#!/usr/bin/env python3
"""Prefill probe against a live MTPLX server (issue-1 measurement surface).

`mtplx bench prefill-ladder` is in-process only and crashes on the qwen4_exp arch
(graphbank eager fallback -> AttributeError: 'DecoderLayer' object has no attribute
'input_layernorm'), so this probe measures the *serve* path instead: it fires a long
prompt per rung, then one small follow-up turn on the same prefix, and reads the engine's
own per-request accounting out of $MTPLX_HOME/logs/request-log-<port>.jsonl.

    python3 scripts/prefill_probe.py --port 9002 --contexts 8k,32k,64k,128k --tag baseline

Row kind is taken from the engine's cached_tokens, never assumed: a rung whose prompt
shares a prefix with an earlier rung reports cache_source=ram and is labelled growth, not
cold. Pass --unique-body when you want every rung genuinely cold-started.

Requires a server already listening on the port (scripts/serve-flash-next-*.sh).
Writes bench/prefill-probe-<port>-<tag>-<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path

MTPLX_HOME = Path(os.environ.get("MTPLX_HOME", Path.home() / ".mtplx"))
CHARS_PER_TOKEN = 3.6  # rough; actual prompt_tokens come back from the engine
SALT = os.environ.get("MTPLX_PROBE_SALT", "mtplx-issue1")

FILLER = (
    "def reconcile_ledger(entries, opening_balance, tolerance):\n"
    "    running = opening_balance\n"
    "    for entry in entries:\n"
    "        running += entry.amount\n"
    "        if abs(running - entry.expected) > tolerance:\n"
    "            raise DriftError(entry.id, running, entry.expected)\n"
    "    return running\n\n"
)


def request(base: str, payload: dict, timeout: float) -> float:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
    return time.monotonic() - start


def filler_text(target_tokens: int) -> str:
    need = int(target_tokens * CHARS_PER_TOKEN)
    return (FILLER * max(1, need // len(FILLER) + 1))[:need]


def unique_filler(target_tokens: int, seed: str) -> str:
    """Same length as filler_text, but every block carries a rung-specific token so no
    earlier rung's prefix can match. Deterministic in `seed`, so A/B runs stay comparable."""
    rng = random.Random(seed)
    need = int(target_tokens * CHARS_PER_TOKEN)
    blocks = need // len(FILLER) + 1
    out = []
    for i in range(blocks):
        out.append(f"// probe-{seed[:8]}-{i:06d}-{rng.randrange(1 << 30):08x}\n" + FILLER)
    return "".join(out)[:need]


def parse_rungs(spec: str) -> list[int]:
    rungs = []
    for raw in spec.split(","):
        raw = raw.strip().lower().rstrip(".")
        if not raw:
            continue
        mult = 1024 if raw.endswith("k") else 1024 * 1024 if raw.endswith("m") else 1
        rungs.append(int(float(raw.rstrip("kmb")) * mult))
    return rungs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--contexts", default="8k,32k,64k,128k")
    ap.add_argument("--followup-tokens", type=int, default=700)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--unique-body", action="store_true",
                    help="make every rung genuinely cold (no cross-rung prefix sharing)")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    log_path = MTPLX_HOME / "logs" / f"request-log-{args.port}.jsonl"
    out_dir = MTPLX_HOME / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"prefill-probe-{args.port}-{args.tag}-{stamp}.json"

    offset = log_path.stat().st_size if log_path.exists() else 0
    probe_rows: list[dict] = []

    for tokens in parse_rungs(args.contexts):
        seed = f"{SALT}-{args.tag}-{tokens}"
        body = unique_filler(tokens, seed) if args.unique_body else filler_text(tokens)
        msgs = [{"role": "user", "content": f"[probe {seed}]\n{body}"}]
        entry = {"target_tokens": tokens, "unique_body": args.unique_body}

        entry["first_wall_s"] = round(request(base, {
            "messages": msgs, "max_tokens": args.max_new_tokens, "stream": False,
        }, args.timeout_s), 2)
        print(f"ctx~{tokens:>7} first     wall={entry['first_wall_s']:8.2f}s", flush=True)

        entry["followup_wall_s"] = round(request(base, {
            "messages": msgs + [{"role": "user", "content": unique_filler(args.followup_tokens, seed + "-fu")}],
            "max_tokens": args.max_new_tokens, "stream": False,
        }, args.timeout_s), 2)
        print(f"ctx~{tokens:>7} follow-up wall={entry['followup_wall_s']:8.2f}s", flush=True)
        probe_rows.append(entry)

    with log_path.open() as handle:
        handle.seek(offset)
        engine_rows = [json.loads(line) for line in handle if line.strip()]

    summary = []
    for r in engine_rows:
        new = r.get("new_prefill_tokens") or 0
        ev = r.get("prompt_eval_time_s") or 0
        cached = r.get("cached_tokens") or 0
        summary.append({
            "request_id": r.get("request_id"),
            "model": r.get("served_model_id") or r.get("model"),
            "kind": "cold" if cached == 0 else "prefix_hit",
            "prompt_tokens": r.get("prompt_tokens"),
            "new_prefill_tokens": new,
            "cached_tokens": cached,
            "cache_source": r.get("cache_source"),
            "prompt_eval_time_s": round(ev, 3) if ev else None,
            "implied_new_tok_s": round(new / ev, 1) if new and ev else None,
            "prefill_tok_s": r.get("prefill_tok_s"),
            "ttft_s": r.get("ttft_s"),
            "decode_tok_s": r.get("decode_tok_s"),
            "target_forward_time_s": r.get("target_forward_time_s"),
            "paged_gqa_sdpa_calls": r.get("paged_gqa_sdpa_calls"),
            "prefill_dense_fallback_calls": r.get("prefill_dense_fallback_calls"),
            "prefill_partitioned_paged_calls": r.get("prefill_partitioned_paged_calls"),
            "prefill_attention_impl": r.get("prefill_attention_impl"),
            "prefill_layout": r.get("prefill_layout"),
            "peak_memory_bytes": r.get("peak_memory_bytes"),
            "active_memory_bytes": r.get("active_memory_bytes"),
            "cache_memory_bytes": r.get("cache_memory_bytes"),
        })

    payload = {
        "salt": SALT, "tag": args.tag, "port": args.port, "host": args.host,
        "unique_body": args.unique_body, "max_new_tokens": args.max_new_tokens,
        "probe": probe_rows, "engine": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"\n{'kind':11}{'prompt':>9}{'new':>8}{'cached':>9}{'eval_s':>9}"
          f"{'new tok/s':>11}{'ttft_s':>8}{'peak GiB':>10}")
    for s in summary:
        print(f"{s['kind']:11}{s['prompt_tokens'] or 0:>9}{s['new_prefill_tokens']:>8}"
              f"{s['cached_tokens']:>9}{(s['prompt_eval_time_s'] or 0):>9.2f}"
              f"{(s['implied_new_tok_s'] or 0):>11.1f}{(s['ttft_s'] or 0):>8.2f}"
              f"{(s['peak_memory_bytes'] or 0) / 2**30:>10.1f}")
    print(f"\nwrote {out_path} ({len(summary)} engine rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
