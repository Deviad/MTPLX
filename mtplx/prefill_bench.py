"""Prompt-prefill ladder benchmark used by v0.1.7 release QA."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hardware import inspect_hardware
from .profiles import DEFAULT_PROFILE_NAME, apply_profile_env, get_profile


DEFAULT_CONTEXTS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
FULL_CONTEXTS = DEFAULT_CONTEXTS + (65536, 131072)
QWEN4_ENV_KEY = "MTPLX_QWEN4_FIXED_M4_VERIFY"
DEFAULT_PROMPT_STYLE = "coding-agent"
LEGACY_PROMPT_STYLE = "legacy-repeat"
PROMPT_STYLE_CHOICES = (DEFAULT_PROMPT_STYLE, LEGACY_PROMPT_STYLE)
DEFAULT_PROMPT_FORMAT = "chat"
RAW_PROMPT_FORMAT = "raw"
PROMPT_FORMAT_CHOICES = (DEFAULT_PROMPT_FORMAT, RAW_PROMPT_FORMAT)
PROFILE_PREFILL_LAYOUT = "profile"
CONTIGUOUS_THEN_REPAGE_LAYOUT = "contiguous-then-repage"
CONTIGUOUS_DENSE_DECODE_LAYOUT = "contiguous-dense-decode"
PAGED_PREFILL_LAYOUT = "paged"
PREFILL_LAYOUT_CHOICES = (
    PROFILE_PREFILL_LAYOUT,
    CONTIGUOUS_THEN_REPAGE_LAYOUT,
    CONTIGUOUS_DENSE_DECODE_LAYOUT,
    PAGED_PREFILL_LAYOUT,
)
PROMPT_POLICY_VERSION = "coding_agent_tail_v2"
UNSAFE_STOCK_CACHE_ONLY_ALLOW_ENV = "MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY"
DEFAULT_SYSTEM_PROMPT = (
    "You are MTPLX, a precise coding agent. Follow the user's instructions, "
    "preserve exact behavior, and prefer production-safe patches."
)
DEFAULT_FINAL_REQUEST = (
    "\n\n# Final user request\n"
    "Write code only. Create a single Python file that behaves like a small "
    "production package for deterministic benchmark runs. No prose outside "
    "code. Use Python 3.11, dataclasses, pathlib, json, argparse, time, "
    "hashlib, statistics, and typing. Keep it compact but complete.\n\n"
    "Implement these sections in order, separated by short comments:\n"
    "1. prompt schema dataclasses and JSONL loader\n"
    "2. validation helpers for required fields, token limits, duplicate ids, "
    "and deterministic hashing\n"
    "3. an LRU cache with get, put, delete, clear, stats, and JSON snapshot "
    "methods\n"
    "4. a rolling metrics window with mean, p50, p90, p95, min, max, and rate "
    "helpers\n"
    "5. benchmark record dataclasses with serialization and summary methods\n"
    "6. a deterministic sampler config object with top_k/top_p/temperature "
    "validation\n"
    "7. a tiny event log writer that appends JSONL rows atomically\n"
    "8. a run registry that stores run metadata, artifacts, git hash, model "
    "path, and environment flags\n"
    "9. a CLI with subcommands validate-prompts, summarize-runs, inspect-cache, "
    "and write-demo\n"
    "10. a small self-test function that exercises every component and returns "
    "a structured dict\n\n"
    "Start now with imports and implement the full file through the CLI "
    "entrypoint.\n"
)


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _unsafe_stock_cache_only_allowed() -> bool:
    return _env_truthy(os.environ.get(UNSAFE_STOCK_CACHE_ONLY_ALLOW_ENV))


@dataclass(frozen=True)
class PromptBuild:
    token_ids: list[int]
    metadata: dict[str, Any]


class UnsafePrefillDiagnosticError(RuntimeError):
    """Raised when a known-risk diagnostic prefill path is requested casually."""


def parse_contexts(value: str | None, *, full: bool = False) -> list[int]:
    if not value:
        return list(FULL_CONTEXTS if full else DEFAULT_CONTEXTS)
    contexts: list[int] = []
    for piece in value.replace(";", ",").split(","):
        raw = piece.strip().lower()
        if not raw:
            continue
        multiplier = 1
        if raw.endswith("k"):
            multiplier = 1024
            raw = raw[:-1]
        contexts.append(max(1, int(float(raw) * multiplier)))
    return contexts


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_prompt_text() -> str:
    base = (
        "You are a coding agent working inside a large Python repository. "
        "Read the following files, preserve exact behavior, and identify the "
        "smallest production-safe patch. Return only implementation notes and "
        "the final patch rationale.\n\n"
    )
    module = (
        "from __future__ import annotations\n\n"
        "import dataclasses\nimport json\nimport time\n"
        "from pathlib import Path\nfrom typing import Any\n\n"
        "@dataclasses.dataclass(frozen=True)\n"
        "class RequestState:\n"
        "    request_id: str\n"
        "    prompt_tokens: int\n"
        "    started_at: float\n"
        "    metadata: dict[str, Any]\n\n"
        "def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:\n"
        "    out = dict(payload)\n"
        "    out.setdefault('created_at', time.time())\n"
        "    out.setdefault('source', 'prefill-ladder')\n"
        "    return out\n\n"
    )
    return base + "\n".join(f"# file_{idx}.py\n{module}" for idx in range(96))


def _legacy_token_ids_for_context(tokenizer: Any, context_tokens: int) -> PromptBuild:
    prompt = _model_prompt_text()
    ids = list(tokenizer.encode(prompt))
    while len(ids) < context_tokens:
        prompt += "\n\n" + _model_prompt_text()
        ids = list(tokenizer.encode(prompt))
    token_ids = [int(token) for token in ids[:context_tokens]]
    return PromptBuild(
        token_ids=token_ids,
        metadata={
            "prompt_policy": "legacy_repeat_hard_truncate",
            "prompt_style": LEGACY_PROMPT_STYLE,
            "prompt_release_valid": False,
            "prompt_text_sha256": _sha256(prompt),
            "prompt_context_tokens": int(context_tokens),
            "prompt_actual_tokens": len(token_ids),
            "prompt_tail_tokens": 0,
            "prompt_filler_tokens": len(token_ids),
            "prompt_tail_preserved": False,
            "prompt_tail_sha256": "",
        },
    )


def _coherent_tail_token_ids_for_context(
    tokenizer: Any,
    context_tokens: int,
    *,
    prompt_tail: str | None = None,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    enable_thinking: bool | None = False,
) -> PromptBuild:
    tail = prompt_tail if prompt_tail is not None else DEFAULT_FINAL_REQUEST
    prompt_format = _normalize_prompt_format(prompt_format)
    tail_ids = _encode_prompt_content(
        tokenizer,
        tail,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
    )
    if not tail_ids:
        raise ValueError("prompt tail must encode to at least one token")

    if len(tail_ids) >= context_tokens:
        token_ids = tail_ids[-context_tokens:]
        return PromptBuild(
            token_ids=token_ids,
            metadata={
                "prompt_policy": PROMPT_POLICY_VERSION,
                "prompt_style": DEFAULT_PROMPT_STYLE,
                "prompt_format": prompt_format,
                "prompt_enable_thinking": enable_thinking,
                "prompt_release_valid": False,
                "prompt_tail_sha256": _sha256(tail),
                "prompt_tail_tokens": len(tail_ids),
                "prompt_tail_preserved": False,
                "prompt_tail_truncated": True,
                "prompt_filler_tokens": 0,
                "prompt_context_tokens": int(context_tokens),
                "prompt_actual_tokens": len(token_ids),
            },
        )

    filler_target = int(context_tokens) - len(tail_ids)
    filler = _model_prompt_text()
    raw_filler_ids = [int(token) for token in tokenizer.encode(filler)]
    filler_ids = raw_filler_ids
    while len(filler_ids) < filler_target:
        filler += "\n\n" + _model_prompt_text()
        filler_ids = [int(token) for token in tokenizer.encode(filler)]
    content = tokenizer.decode(filler_ids[:filler_target]) + tail
    token_ids = _encode_prompt_content(
        tokenizer,
        content,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
    )
    while len(token_ids) < context_tokens:
        filler += "\n\n" + _model_prompt_text()
        filler_ids = [int(token) for token in tokenizer.encode(filler)]
        filler_target += max(1, len(raw_filler_ids) // 4)
        content = tokenizer.decode(filler_ids[:filler_target]) + tail
        token_ids = _encode_prompt_content(
            tokenizer,
            content,
            prompt_format=prompt_format,
            enable_thinking=enable_thinking,
        )
    head_trimmed_tokens = max(0, len(token_ids) - int(context_tokens))
    token_ids = token_ids[-context_tokens:]
    return PromptBuild(
        token_ids=token_ids,
        metadata={
            "prompt_policy": PROMPT_POLICY_VERSION,
            "prompt_style": DEFAULT_PROMPT_STYLE,
            "prompt_format": prompt_format,
            "prompt_enable_thinking": enable_thinking,
            "prompt_release_valid": True,
            "prompt_tail_sha256": _sha256(tail),
            "prompt_tail_tokens": len(tail_ids),
            "prompt_tail_preserved": True,
            "prompt_tail_truncated": False,
            "prompt_filler_tokens": filler_target,
            "prompt_head_trimmed_tokens": head_trimmed_tokens,
            "prompt_context_tokens": int(context_tokens),
            "prompt_actual_tokens": len(token_ids),
            "prompt_filler_sha256": _sha256(filler),
        },
    )


def _normalize_prompt_format(prompt_format: str) -> str:
    normalized = (prompt_format or DEFAULT_PROMPT_FORMAT).strip().lower().replace("_", "-")
    if normalized not in PROMPT_FORMAT_CHOICES:
        raise ValueError(
            f"unknown prompt format {prompt_format!r}; expected one of: "
            + ", ".join(PROMPT_FORMAT_CHOICES)
        )
    return normalized


def _normalize_prefill_layout(prefill_layout: str | None) -> str:
    normalized = (
        prefill_layout or PROFILE_PREFILL_LAYOUT
    ).strip().lower().replace("_", "-")
    if normalized in {"", "default"}:
        normalized = PROFILE_PREFILL_LAYOUT
    if normalized not in PREFILL_LAYOUT_CHOICES:
        raise ValueError(
            f"unknown prefill layout {prefill_layout!r}; expected one of: "
            + ", ".join(PREFILL_LAYOUT_CHOICES)
        )
    return normalized


def _prefill_layout_env_value(prefill_layout: str) -> str | None:
    normalized = _normalize_prefill_layout(prefill_layout)
    if normalized == PROFILE_PREFILL_LAYOUT:
        return None
    return normalized.replace("-", "_")


def _apply_prefill_layout_override(prefill_layout: str) -> str | None:
    value = _prefill_layout_env_value(prefill_layout)
    if value is not None:
        os.environ["MTPLX_SUSTAINED_PREFILL_LAYOUT"] = value
    return value


def _apply_paged_attention_impl_override(args: Any) -> str:
    impl = str(getattr(args, "paged_attn_impl", "") or "").strip().lower()
    impl = impl.replace("-", "_")
    if impl:
        os.environ["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] = impl
    return impl


def _apply_mtp_history_policy_override(args: Any) -> str:
    policy = str(getattr(args, "mtp_history_policy", "") or "").strip().lower()
    policy = policy.replace("-", "_")
    if policy:
        os.environ["MTPLX_MTP_HISTORY_POLICY"] = policy
    return policy


def _apply_mtp_history_window_override(args: Any) -> int | None:
    raw = getattr(args, "mtp_history_window", None)
    if raw is None:
        return None
    window = int(raw)
    if window <= 0:
        raise ValueError("--mtp-history-window must be positive")
    os.environ["MTPLX_MTP_HISTORY_LAST_WINDOW"] = str(window)
    return window


def _apply_prefill_cache_cleanup_override(args: Any) -> bool:
    enabled = bool(getattr(args, "prefill_cache_cleanup", False))
    disabled = bool(getattr(args, "no_prefill_cache_cleanup", False))
    if enabled and disabled:
        raise ValueError(
            "--prefill-cache-cleanup and --no-prefill-cache-cleanup cannot be "
            "used together"
        )
    if disabled:
        os.environ["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] = "0"
        return True
    if enabled:
        os.environ["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] = "1"
    return enabled


def _apply_prefill_cache_cleanup_every_override(args: Any) -> int | None:
    raw = getattr(args, "prefill_cache_cleanup_every", None)
    if raw is None:
        return None
    raw_text = str(raw).strip().lower()
    if raw_text == "auto":
        os.environ["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] = "auto"
        return None
    every = int(raw_text)
    if every <= 0:
        raise ValueError("--prefill-cache-cleanup-every must be positive")
    os.environ["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] = str(every)
    return every


def _apply_batch_target_arrays_override(args: Any) -> str | None:
    if not bool(getattr(args, "no_batch_target_arrays", False)):
        return None
    os.environ["MTPLX_BATCH_TARGET_ARRAYS"] = "0"
    return "0"


def _apply_prefill_chunk_size_override(args: Any) -> int | None:
    raw = getattr(args, "prefill_chunk_size", None)
    if raw is None:
        return None
    chunk_size = int(raw)
    if chunk_size <= 0:
        raise ValueError("--prefill-chunk-size must be positive")
    os.environ["MTPLX_PREFILL_CHUNK_SIZE"] = str(chunk_size)
    return chunk_size


def _apply_clear_cache_every_override(args: Any) -> int | None:
    raw = getattr(args, "clear_cache_every", None)
    if raw is None:
        return None
    every = int(raw)
    if every < 0:
        raise ValueError("--clear-cache-every must be non-negative")
    os.environ["MTPLX_CLEAR_CACHE_EVERY"] = str(every)
    return every


def _apply_defer_verify_hidden_override(args: Any) -> str | None:
    enable = bool(getattr(args, "defer_verify_hidden_eval", False))
    disable = bool(getattr(args, "no_defer_verify_hidden_eval", False))
    if enable and disable:
        raise ValueError(
            "--defer-verify-hidden-eval and --no-defer-verify-hidden-eval "
            "cannot be used together"
        )
    if enable:
        os.environ["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] = "1"
        return "1"
    if disable:
        os.environ["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] = "0"
        return "0"
    return None


def _apply_verify_hidden_mode_override(args: Any) -> str | None:
    raw = str(getattr(args, "verify_hidden_mode", "") or "").strip().lower()
    mode = raw.replace("-", "_")
    if not mode:
        return None
    os.environ["MTPLX_VERIFY_HIDDEN_MODE"] = mode
    return mode


def _apply_prefill_stock_cache_only_override(args: Any) -> bool:
    enabled = bool(getattr(args, "prefill_stock_cache_only", False))
    if enabled:
        if not _unsafe_stock_cache_only_allowed():
            raise UnsafePrefillDiagnosticError(
                "--prefill-stock-cache-only is an unsafe diagnostic path after "
                "a 64k M5 Max watchdog panic. Set "
                f"{UNSAFE_STOCK_CACHE_ONLY_ALLOW_ENV}=1 to run it explicitly."
            )
        os.environ["MTPLX_PREFILL_STOCK_CACHE_ONLY"] = "1"
    return enabled


def _encode_prompt_content(
    tokenizer: Any,
    content: str,
    *,
    prompt_format: str,
    enable_thinking: bool | None,
) -> list[int]:
    prompt_format = _normalize_prompt_format(prompt_format)
    if prompt_format == RAW_PROMPT_FORMAT:
        return [int(token) for token in tokenizer.encode(content)]
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("Tokenizer does not expose apply_chat_template")
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return [
        int(token)
        for token in tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
    ]


def _prompt_build_for_context(
    tokenizer: Any,
    context_tokens: int,
    *,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    prompt_tail: str | None = None,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    enable_thinking: bool | None = False,
) -> PromptBuild:
    style = (prompt_style or DEFAULT_PROMPT_STYLE).strip().lower().replace("_", "-")
    if style == LEGACY_PROMPT_STYLE:
        return _legacy_token_ids_for_context(tokenizer, context_tokens)
    if style != DEFAULT_PROMPT_STYLE:
        raise ValueError(
            f"unknown prompt style {prompt_style!r}; expected one of: "
            + ", ".join(PROMPT_STYLE_CHOICES)
        )
    return _coherent_tail_token_ids_for_context(
        tokenizer,
        context_tokens,
        prompt_tail=prompt_tail,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
    )


def _token_ids_for_context(
    tokenizer: Any,
    context_tokens: int,
    *,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    prompt_tail: str | None = None,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    enable_thinking: bool | None = False,
) -> list[int]:
    return _prompt_build_for_context(
        tokenizer,
        context_tokens,
        prompt_style=prompt_style,
        prompt_tail=prompt_tail,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
    ).token_ids


def _load_prompt_tail(args: Any) -> str:
    tail_file = getattr(args, "prompt_tail_file", None)
    if tail_file:
        return Path(tail_file).read_text(encoding="utf-8")
    tail = getattr(args, "prompt_tail", None)
    if tail:
        return str(tail)
    return DEFAULT_FINAL_REQUEST


def _prompt_release_valid(prompt_style: str, prompt_tail: str) -> bool:
    if prompt_style == LEGACY_PROMPT_STYLE:
        return False
    return bool(prompt_tail.strip())


def _recommended_prefill_qa_commands(
    *,
    model: str,
    profile: str,
    prompt_style: str,
    prompt_format: str,
    prefill_layout: str,
    max_tokens: int,
) -> list[str]:
    layout_arg = ""
    if _normalize_prefill_layout(prefill_layout) != PROFILE_PREFILL_LAYOUT:
        layout_arg = f"--prefill-layout {shlex.quote(prefill_layout)} "
    base = (
        "uv run python -m mtplx.cli bench prefill-ladder "
        f"--model {shlex.quote(model)} "
        f"--profile {shlex.quote(profile)} --max "
        f"--prompt-style {shlex.quote(prompt_style)} "
        f"--prompt-format {shlex.quote(prompt_format)} "
        f"{layout_arg}"
        "--disable-thinking "
        f"--max-tokens {int(max_tokens)} "
    )
    return [
        base
        + "--contexts 16384,32768 "
        + "--output benchmarks/results/prefill-fixed-m5max-local-16k-32k-coherent-tail.json",
        base
        + "--contexts 65536 "
        + "--output benchmarks/results/prefill-fixed-m5max-local-64k-coherent-tail.json",
        base
        + "--contexts 131072 "
        + "--output benchmarks/results/prefill-fixed-m5max-local-128k-coherent-tail.json",
    ]


def _env_snapshot() -> dict[str, str]:
    keys = (
        "MTPLX_PREFILL_CHUNK_SIZE",
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
        "MTPLX_PREFILL_CHUNK_SIZE_REPAGE",
        "MTPLX_PREFILL_CHUNK_CACHE_CLEANUP",
        "MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY",
        "MTPLX_PREFILL_OMLX_EXTERNAL",
        "MTPLX_PREFILL_EXTERNAL_EMIT_LOGITS",
        "MTPLX_PREFILL_STOCK_CACHE_ONLY",
        UNSAFE_STOCK_CACHE_ONLY_ALLOW_ENV,
        "MTPLX_CLEAR_CACHE_EVERY",
        "MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD",
        "MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT",
        "MTPLX_SUSTAINED_PREFILL",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT",
        "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT",
        "MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS",
        "MTPLX_DEFER_VERIFY_HIDDEN_EVAL",
        "MTPLX_VERIFY_HIDDEN_MODE",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD",
        "MTPLX_LONG_CONTEXT_MTP_DEPTH",
        "MTPLX_VLLM_METAL_PAGED_ATTN",
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL",
        "MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q",
        "MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD",
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE",
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT",
        "MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE",
        "MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE",
        "MTPLX_ASSERT_NO_LARGE_Q_SPLIT_FALLBACK",
        "MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS",
        "MTPLX_PREFILL_ROUTE_TRACE",
        "MTPLX_LAZY_VERIFY_LOGITS",
        "MTPLX_BATCH_TARGET_ARRAYS",
        "MTPLX_LAZY_MTP_HISTORY_APPEND",
        "MTPLX_MTP_HISTORY_POLICY",
        "MTPLX_MTP_HISTORY_LAST_WINDOW",
        "MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD",
        "MTPLX_DROP_EVENTS",
        "MTPLX_SKIP_VERIFY_SNAPSHOT",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _stats_value(stats: Any, key: str, default: Any = 0) -> Any:
    if isinstance(stats, dict):
        return stats.get(key, default)
    return getattr(stats, key, default)


def _sync_and_clear_cache_between_contexts() -> float:
    """Drain MLX work and clear reusable buffers between ladder rows.

    This is deliberately benchmark-harness hygiene, not timed model work. It
    mirrors oMLX's safety rule: synchronize first, then clear, so we do not
    release buffers still referenced by in-flight Metal command buffers.
    """
    started = time.perf_counter()
    try:
        import mlx.core as mx

        try:
            mx.synchronize()
        except RuntimeError:
            pass
        mx.clear_cache()
    finally:
        gc.collect()
    return time.perf_counter() - started


def _int_stats_or_env(
    stats: Any,
    stats_key: str,
    env_key: str,
    *,
    default: int = 0,
) -> int:
    value = _stats_value(stats, stats_key, None)
    if value is None:
        value = os.environ.get(env_key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_int_value(env_key: str, *, default: int = 0) -> int:
    try:
        return int(os.environ.get(env_key) or default)
    except (TypeError, ValueError):
        return default


def _row_from_output(
    *,
    context_tokens: int,
    output: Any,
    request_started_s: float,
    first_token_s: float | None,
) -> dict[str, Any]:
    stats = output.stats
    generated = int(_stats_value(stats, "generated_tokens", len(output.tokens)))
    prompt_eval = float(_stats_value(stats, "prompt_eval_time_s", 0.0) or 0.0)
    elapsed = float(_stats_value(stats, "elapsed_s", 0.0) or 0.0)
    decode_elapsed = max(0.0, elapsed - prompt_eval)
    owned = _stats_value(stats, "owned_attn_kv", {}) or {}
    prompt_tps = float(_stats_value(stats, "prompt_tps", 0.0) or 0.0)
    if prompt_tps <= 0 and prompt_eval > 0:
        prompt_tps = context_tokens / prompt_eval
    return {
        "context_tokens": int(context_tokens),
        "prompt_tps": prompt_tps,
        "pp_tps": prompt_tps,
        "ttft_s": (
            max(0.0, float(first_token_s) - request_started_s)
            if first_token_s is not None
            else None
        ),
        "decode_tok_s": generated / decode_elapsed if decode_elapsed > 0 else 0.0,
        "generated_tokens": generated,
        "speculative_depth": int(_stats_value(stats, "speculative_depth", 0) or 0),
        "requested_speculative_depth": int(
            _stats_value(stats, "requested_speculative_depth", 0) or 0
        ),
        "long_context_mtp_depth_policy": dict(
            _stats_value(stats, "long_context_mtp_depth_policy", {}) or {}
        ),
        "accepted_drafts": int(_stats_value(stats, "accepted_drafts", 0) or 0),
        "drafted_tokens": int(_stats_value(stats, "drafted_tokens", 0) or 0),
        "draft_acceptance_rate": (
            float(_stats_value(stats, "accepted_drafts", 0) or 0)
            / float(_stats_value(stats, "drafted_tokens", 0) or 1)
        ),
        "verify_calls": int(_stats_value(stats, "verify_calls", 0) or 0),
        "verify_time_s": float(_stats_value(stats, "verify_time_s", 0.0) or 0.0),
        "verify_forward_time_s": float(
            _stats_value(stats, "verify_forward_time_s", 0.0) or 0.0
        ),
        "verify_eval_time_s": float(
            _stats_value(stats, "verify_eval_time_s", 0.0) or 0.0
        ),
        "verify_logits_eval_time_s": float(
            _stats_value(stats, "verify_logits_eval_time_s", 0.0) or 0.0
        ),
        "verify_hidden_eval_time_s": float(
            _stats_value(stats, "verify_hidden_eval_time_s", 0.0) or 0.0
        ),
        "verify_joint_eval_time_s": float(
            _stats_value(stats, "verify_joint_eval_time_s", 0.0) or 0.0
        ),
        "verify_eval_unattributed_time_s": float(
            _stats_value(stats, "verify_eval_unattributed_time_s", 0.0) or 0.0
        ),
        "verify_hidden_mode": str(
            _stats_value(stats, "verify_hidden_mode", "") or ""
        ),
        "draft_time_s": float(_stats_value(stats, "draft_time_s", 0.0) or 0.0),
        "repair_time_s": float(_stats_value(stats, "repair_time_s", 0.0) or 0.0),
        "target_forward_time_s": float(
            _stats_value(stats, "target_forward_time_s", 0.0) or 0.0
        ),
        "elapsed_s": elapsed,
        "decode_elapsed_s": decode_elapsed,
        "peak_memory_gb": float(_stats_value(stats, "peak_memory_bytes", 0) or 0)
        / (1024**3),
        "prompt_eval_time_s": prompt_eval,
        "prompt_target_prefill_time_s": float(
            _stats_value(stats, "prompt_target_prefill_time_s", 0.0) or 0.0
        ),
        "prompt_mtp_history_time_s": float(
            _stats_value(stats, "prompt_mtp_history_time_s", 0.0) or 0.0
        ),
        "prompt_repair_time_s": float(
            _stats_value(stats, "prompt_repair_time_s", 0.0) or 0.0
        ),
        "prompt_suffix_time_s": float(
            _stats_value(stats, "prompt_suffix_time_s", 0.0) or 0.0
        ),
        "prompt_repage_time_s": float(
            _stats_value(stats, "prompt_repage_time_s", 0.0) or 0.0
        ),
        "prompt_eval_breakdown_complete": bool(
            _stats_value(stats, "prompt_eval_breakdown_complete", False)
        ),
        "prompt_target_prefill_tok_s": float(
            _stats_value(stats, "prompt_target_prefill_tok_s", 0.0) or 0.0
        ),
        "prompt_mtp_history_tok_s": float(
            _stats_value(stats, "prompt_mtp_history_tok_s", 0.0) or 0.0
        ),
        "prefill_chunk_cache_cleanup_enabled": bool(
            _stats_value(stats, "prefill_chunk_cache_cleanup_enabled", False)
        ),
        "prefill_chunk_cache_cleanup_every": int(
            _stats_value(stats, "prefill_chunk_cache_cleanup_every", 1) or 1
        ),
        "prefill_chunk_cache_cleanup_events": int(
            _stats_value(stats, "prefill_chunk_cache_cleanup_events", 0) or 0
        ),
        "prefill_stock_cache_only_enabled": bool(
            _stats_value(stats, "prefill_stock_cache_only_enabled", False)
        ),
        "prefill_stock_cache_only_calls": int(
            _stats_value(stats, "prefill_stock_cache_only_calls", 0) or 0
        ),
        "prefill_omlx_external_enabled": bool(
            _stats_value(stats, "prefill_omlx_external_enabled", False)
        ),
        "prefill_omlx_external_calls": int(
            _stats_value(stats, "prefill_omlx_external_calls", 0) or 0
        ),
        "prefill_external_emit_logits_enabled": bool(
            _stats_value(stats, "prefill_external_emit_logits_enabled", True)
        ),
        "prefill_external_cache_only_calls": int(
            _stats_value(stats, "prefill_external_cache_only_calls", 0) or 0
        ),
        "mtp_history_policy": str(_stats_value(stats, "mtp_history_policy", "") or ""),
        "mtp_history_window_tokens": int(
            _stats_value(stats, "mtp_history_window_tokens", 0) or 0
        ),
        "mtp_history_position_base": int(
            _stats_value(stats, "mtp_history_position_base", 0) or 0
        ),
        "large_q_split_sdpa_fallback_calls": int(
            _stats_value(stats, "large_q_split_sdpa_fallback_calls", 0) or 0
        ),
        "large_q_split_sdpa_fallback_calls_by_phase": dict(
            _stats_value(stats, "large_q_split_sdpa_fallback_calls_by_phase", {}) or {}
        ),
        "prefill_large_q_split_sdpa_fallback_calls": int(
            _stats_value(stats, "prefill_large_q_split_sdpa_fallback_calls", 0) or 0
        ),
        "partitioned_paged_calls": int(
            _stats_value(stats, "partitioned_paged_calls", 0) or 0
        ),
        "partitioned_paged_calls_by_phase": dict(
            _stats_value(stats, "partitioned_paged_calls_by_phase", {}) or {}
        ),
        "prefill_partitioned_paged_calls": int(
            _stats_value(stats, "prefill_partitioned_paged_calls", 0) or 0
        ),
        "paged_attention_large_q_path": str(
            _stats_value(stats, "paged_attention_large_q_path", "") or ""
        ),
        "prefill_route": str(_stats_value(stats, "prefill_route", "") or ""),
        "prefill_attention_impl": str(
            _stats_value(stats, "prefill_attention_impl", "unrecorded") or "unrecorded"
        ),
        "prefill_layout": str(_stats_value(stats, "prefill_layout", "") or ""),
        "paged_attention_bailouts_by_phase_reason": dict(
            _stats_value(stats, "paged_attention_bailouts_by_phase_reason", {}) or {}
        ),
        "effective_prefill_chunk_size": _int_stats_or_env(
            stats,
            "prefill_chunk_size",
            "MTPLX_PREFILL_CHUNK_SIZE",
        ),
        "effective_partition_size": _env_int_value(
            "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE"
        ),
        "effective_large_q_chunk_size": _env_int_value(
            "MTPLX_VLLM_METAL_PAGED_LARGE_Q_CHUNK_SIZE"
        ),
        "effective_large_q_kv_chunk_size": _env_int_value(
            "MTPLX_VLLM_METAL_PAGED_LARGE_Q_KV_CHUNK_SIZE"
        ),
        "owned_attn_kv": owned,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("MTPLX Prefill Ladder")
    print("Context | Prompt TPS | Decode TPS | Gen Tokens | TTFT | Memory | Fallback | Partitioned")
    print("--------|------------|------------|------------|------|--------|----------|------------")
    for row in rows:
        ctx = row["context_tokens"]
        label = f"{ctx // 1024}k" if ctx >= 1024 and ctx % 1024 == 0 else str(ctx)
        ttft = row["ttft_s"]
        print(
            f"{label:>7} | "
            f"{row['prompt_tps']:>10.1f} | "
            f"{row['decode_tok_s']:>10.1f} | "
            f"{row['generated_tokens']:>10} | "
            f"{(ttft if ttft is not None else 0.0):>4.1f}s | "
            f"{row['peak_memory_gb']:>5.1f}GB | "
            f"{row['large_q_split_sdpa_fallback_calls']:>8} | "
            f"{row['partitioned_paged_calls']:>10}"
        )


def _ladder_profile(args: Any) -> Any:
    """Profile the ladder runs under: the launch rule, unless --profile is set.

    The raw ``DEFAULT_PROFILE_NAME`` fallback here was the last bench-lane
    sustained side door: without an explicit --profile the flagships silently
    measured the slow profile instead of their turbo launch default. The
    import is lazy in both directions (commands.public imports this module
    inside its bench dispatcher), so there is no module cycle.
    """

    requested = getattr(args, "profile", None)
    if requested:
        return get_profile(str(requested))
    from mtplx.commands.public import _resolved_default_profile_name

    return get_profile(_resolved_default_profile_name(args))


def _ladder_verify_route(model: str) -> tuple[str, str]:
    """The verify strategy/core this pack's family can actually run.

    The ladder measured ``capture_commit`` unconditionally, which is a qwen3-next
    structure lane: the server coerces it to ``batched`` for qwen4_exp packs
    (``openai._coerce_family_verify_strategy``) because the capture stack
    introspects the qwen3-next layer layout. Measured on the Flash-Next pack, the
    mismatch produced qwen4-shaped capture rows with no ``conv_states`` and then
    ``KeyError: 'conv_states'`` inside the generic commit.
    """
    from .qwen4_fixed_verify import family_verify_strategy

    strategy = family_verify_strategy(model, "capture_commit")
    if strategy == "capture_commit":
        return "capture_commit", "linear-gdn-from-conv-tape"
    # batched is the family's base lane; "stock" is generate_mtpk's own default,
    # so the ladder asks for what `mtplx serve` would run rather than a custom core.
    return strategy, "stock"


def _apply_family_verify_lane_override(model: str) -> str | None:
    """Install the verify lane the pack's own family requires, if any.

    ``server/openai.py`` resolves the qwen4 lane env from the pack's ``config.json``;
    the ladder built its environment from the profile alone, so a ``qwen4_exp`` trunk
    reached the generic hybrid capture — written for the qwen3_5 layer vocabulary — and
    died inside the first captured forward with an ``AttributeError`` that read as
    damaged model code. Keyed on ``model_type`` because this family rides the generic
    ``native_mtp`` descriptor, not its own. Operator env still wins: this is a
    ``setdefault``, so a deliberately disabled lane stays disabled.

    Returns the value now in effect, or ``None`` when no lane applies (including a
    bare HF id or a directory without ``config.json``, which is not an error here).
    """

    from .qwen4_fixed_verify import QWEN4_FIXED_M4_VERIFY_ENV

    config_path = Path(str(model)).expanduser() / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if str(config.get("model_type") or "") != "qwen4_exp":
        return None
    os.environ.setdefault(QWEN4_FIXED_M4_VERIFY_ENV, "1")
    # The batched coercion alone is not the family lane. `_coerce_family_strategy`
    # (openai.py) documents the pairing: qwen4_exp verifies `batched` and rides
    # `MTPLX_FAMILY_CAPTURE_COMMIT` for repair-free rollback. Without it, a rejected
    # window has no snapshot to fall back on -- profiles default
    # MTPLX_SKIP_VERIFY_SNAPSHOT=1 -- so generation raises
    # "capture commit failed after MTPLX_SKIP_VERIFY_SNAPSHOT=1" (measured on the
    # Flash-Next pack 2026-09-06, at ladder row 2k).
    os.environ.setdefault("MTPLX_FAMILY_CAPTURE_COMMIT", "1")
    return os.environ[QWEN4_FIXED_M4_VERIFY_ENV]


# --- serve harness: measure a live server instead of loading the pack here -------
#
# The in-process ladder cannot run some families at all: enabling a pack's own
# verify lane needs the server's cache containers, so `runtime.load()` reaches
# `unsupported_container:ArraysCache` (graphbank.py) on qwen4_exp. Hand-mirroring
# the server's env in the bench reaches a second wall, so the ladder can instead
# drive a real server and measure what the product actually does.

SERVER_HARNESS = "direct-http"
BENCH_DEFAULT_URL = "http://127.0.0.1:8000"
BENCH_DEFAULT_PORT = 8041
SERVER_TIMEOUT_S = 1800.0
MIN_SERVER_DECODE_TOKENS = 8


def _ladder_server_url(args: Any) -> str | None:
    """Base URL of the server under test, or ``None`` when the harness is in-process.

    Selected by ``--harness direct-http``, the bench vocabulary for driving a server
    over HTTP. An explicit ``--port`` wins because a second MTPLX instance on its own
    port is the common case; otherwise ``--url`` is used as given, and that flag
    already defaults to MTPLX's own default port 8000.
    """

    if str(getattr(args, "harness", "auto") or "auto").strip().lower() != SERVER_HARNESS:
        return None
    port = getattr(args, "port", None)
    if port is not None and int(port) != BENCH_DEFAULT_PORT:
        return f"http://127.0.0.1:{int(port)}"
    url = str(getattr(args, "url", "") or "").strip()
    return (url or BENCH_DEFAULT_URL).rstrip("/")


def _ladder_tokenizer(model: str) -> Any:
    """Tokenizer only — the serve harness must never allocate the pack's weights.

    Reuses the runtime's resilient tokenizer loader (it carries the `tokenizer.json`
    fallback for packs whose `tokenizer_config.json` a strict `AutoTokenizer` rejects)
    instead of re-implementing that recovery here.
    """

    path = Path(str(model)).expanduser()
    config_path = path / "config.json"
    if config_path.exists():
        from .runtime import _load_tokenizer_resilient

        return _load_tokenizer_resilient(
            path, json.loads(config_path.read_text(encoding="utf-8"))
        )
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(str(model))


def _ladder_v1(base_url: str) -> str:
    """OpenAI-compatibility root of a server URL given without it.

    ``bench --url`` defaults to ``http://127.0.0.1:8000`` with no ``/v1``, while the
    endpoints live under ``/v1``; appending here keeps both spellings working and the
    404 that a missing segment produces out of the receipts.
    """

    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def _ladder_unreachable(base_url: str, exc: Exception) -> RuntimeError:
    """Split "no server" from "the server answered and refused".

    `HTTPError` subclasses `URLError`, so catching them together reported a 400
    refusal as an unreachable host — measured while first running the harness at
    the pack's full context, where the server said why and this said nothing.
    """

    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001 - a broken body must not mask the status
            detail = ""
        return RuntimeError(
            f"prefill ladder was refused by the server at {base_url}: HTTP {exc.code}"
            f"{': ' + detail if detail else ''}"
        )
    return RuntimeError(
        f"prefill ladder cannot reach the server at {base_url}: "
        f"{type(exc).__name__}: {exc}. Start it (mtplx serve --port ...) or pass the "
        "right --url/--port."
    )


def _ladder_served_target(base_url: str, *, timeout_s: float) -> dict[str, Any]:
    """Served id and advertised window, so a receipt names what it measured.

    The window is what the in-process path checks before loading (#261 / F7:
    "refuse contexts beyond the model's context window instead of silently
    benchmarking past the trained window"); in serve mode the same discipline has
    to ask the server, because the server is the thing that will refuse."""

    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"{_ladder_v1(base_url)}/models", timeout=timeout_s
        ) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    models = data.get("data") or []
    if not models or not isinstance(models[0], dict):
        return {}
    model = models[0]
    target: dict[str, Any] = {}
    if model.get("id"):
        target["model_id"] = str(model["id"])
    for key in ("context_length", "max_context_length", "model_max_length"):
        if isinstance(model.get(key), int) and model[key] > 0:
            target["context_length"] = int(model[key])
            break
    return target


def _ladder_server_row(
    base_url: str,
    *,
    context_tokens: int,
    text: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    timeout_s: float,
) -> dict[str, Any]:
    """One streamed chat completion, reported with the ladder's row vocabulary.

    MTPLX puts the final ``usage`` on the stream without ``stream_options``, which is
    what makes a single request enough: first-delta time gives TTFT, and
    ``usage.prompt_tokens_details.cached_tokens`` separates re-prefill from reuse.
    """

    import time
    import urllib.request

    request_body = {
        "model": model_id,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "seed": int(seed),
        "stream": True,
    }
    request = urllib.request.Request(
        f"{_ladder_v1(base_url)}/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_s = time.perf_counter()
    first_token_s: float | None = None
    usage: dict[str, Any] = {}
    import urllib.error

    try:
        opened = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as exc:
        raise _ladder_unreachable(base_url, exc) from exc
    except urllib.error.URLError as exc:
        raise _ladder_unreachable(base_url, exc) from exc
    with opened as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except ValueError:
                continue
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                emitted = delta.get("content") or delta.get("reasoning_content")
                if emitted and first_token_s is None:
                    first_token_s = time.perf_counter()
    total_s = time.perf_counter() - started_s
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    generated = int(usage.get("completion_tokens") or 0)
    new_prefill = max(0, prompt_tokens - cached_tokens)
    ttft_s = (first_token_s - started_s) if first_token_s is not None else None
    prompt_tps = (
        new_prefill / ttft_s if ttft_s and ttft_s > 0 and new_prefill > 0 else 0.0
    )
    decode_elapsed_s = max(0.0, total_s - (ttft_s or 0.0))
    return {
        "context_tokens": int(context_tokens),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "new_prefill_tokens": new_prefill,
        "context_delta_tokens": prompt_tokens - int(context_tokens),
        "prompt_tps": prompt_tps,
        "pp_tps": prompt_tps,
        # Includes the first token's sampling: the ladder's own `ttft_s` is the same
        # kind of bound, so the two stay comparable instead of quietly different.
        "ttft_s": ttft_s,
        "wall_s": total_s,
        # Decode needs enough samples to mean anything: with max_tokens=4 a
        # 4-token tail divided by the residual window reads 1410 tok/s on a
        # 512-token context (measured), so short rows report no rate at all.
        "decode_tok_s": (
            (max(0, generated - 1) / decode_elapsed_s)
            if decode_elapsed_s > 0 and generated >= MIN_SERVER_DECODE_TOKENS
            else None
        ),
        "generated_tokens": generated,
        "measure": "serve",
    }


def _run_prefill_ladder_against_server(
    args: Any,
    *,
    payload: dict[str, Any],
    base_url: str,
    contexts: list[int],
    prompt_style: str,
    prompt_format: str,
    prompt_tail: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    """The ladder's prompt policy, delivered to a server instead of a local runtime."""

    model = str(payload.get("model") or getattr(args, "model", "") or "")
    target = _ladder_served_target(base_url, timeout_s=10.0)
    served_id = str(target.get("model_id") or model)
    window = int(target.get("context_length") or 0)
    max_tokens = int(getattr(args, "max_tokens", 128) or 128)
    over = [int(c) for c in contexts if window and int(c) + max_tokens > window]
    if over:
        raise RuntimeError(
            f"prefill ladder contexts {over} do not fit the server at {base_url}, which "
            f"advertises {window} tokens (asked {max_tokens} more to generate). Drop them "
            "with --contexts or raise the server's --context-window."
        )
    payload["harness"] = SERVER_HARNESS
    payload["in_process_model_load"] = False
    payload["server"] = {"url": base_url, **target}
    tokenizer = _ladder_tokenizer(model)
    payload.setdefault("rows", [])
    seed_base = int(getattr(args, "seed", None) or 0)
    vary_seed_by_context = bool(getattr(args, "vary_seed_by_context", False))
    for index, context_tokens in enumerate(contexts):
        build = _prompt_build_for_context(
            tokenizer,
            int(context_tokens),
            prompt_style=prompt_style,
            prompt_tail=prompt_tail,
            prompt_format=prompt_format,
            enable_thinking=enable_thinking,
        )
        row = _ladder_server_row(
            base_url,
            context_tokens=int(context_tokens),
            text=tokenizer.decode(build.token_ids),
            model_id=served_id,
            max_tokens=max_tokens,
            temperature=float(getattr(args, "temperature", 0.6) or 0.6),
            top_p=float(getattr(args, "top_p", 0.95) or 0.95),
            top_k=int(getattr(args, "top_k", 20) or 20),
            seed=(seed_base + index if vary_seed_by_context else seed_base),
            timeout_s=SERVER_TIMEOUT_S,
        )
        row.update(build.metadata)
        row["requested_prefill_layout"] = payload.get("prefill_layout", {}).get("requested")
        # The engine names these two only in its own per-request row (see
        # PUBLIC_MTPLX_STATS_KEYS), which the serve harness cannot read over HTTP.
        # Keys stay present as null so a sweep can tell "serve mode" from "no row".
        row["prefill_attention_impl"] = None
        row["prefill_layout"] = None
        payload["rows"].append(row)
    # The layout sweep this key would carry is an in-process knob; in serve mode the
    # server under test owns it, so the key stays present and empty rather than absent
    # (JSON consumers read it unconditionally).
    payload["recommended_plugged_in_commands"] = []
    payload["serve_harness_note"] = (
        "profile env, layout and allocator settings belong to the server under test; "
        "read them from its /mtplx/settings, not from this payload. Cold-prefill rows "
        "require a fresh server too: the session bank survives a request, so a repeated "
        "context reports cached_tokens ~= prompt_tokens and a prefill rate of 0 (measured: "
        "32777/32777 cached, ttft 0.30s on a second run)"
    )
    return payload


def run_prefill_ladder(args: Any) -> dict[str, Any]:
    contexts = parse_contexts(getattr(args, "contexts", None), full=bool(getattr(args, "full", False)))
    profile = _ladder_profile(args)
    prompt_style = str(getattr(args, "prompt_style", None) or DEFAULT_PROMPT_STYLE)
    prompt_format = _normalize_prompt_format(
        str(getattr(args, "prompt_format", None) or DEFAULT_PROMPT_FORMAT)
    )
    prefill_layout = _normalize_prefill_layout(
        str(getattr(args, "prefill_layout", None) or PROFILE_PREFILL_LAYOUT)
    )
    prefill_layout_env_value = _prefill_layout_env_value(prefill_layout)
    enable_thinking = False
    if bool(getattr(args, "enable_thinking", False)):
        enable_thinking = True
    if bool(getattr(args, "disable_thinking", False)):
        enable_thinking = False
    prompt_tail = _load_prompt_tail(args)
    release_valid_prompt = _prompt_release_valid(prompt_style, prompt_tail)
    model = str(getattr(args, "model", ""))
    profile_env = profile.env_dict()
    if prefill_layout_env_value is not None:
        profile_env["MTPLX_SUSTAINED_PREFILL_LAYOUT"] = prefill_layout_env_value
    paged_attn_impl_requested = str(getattr(args, "paged_attn_impl", "") or "").strip().lower().replace("-", "_")
    mtp_history_policy_requested = str(getattr(args, "mtp_history_policy", "") or "").strip().lower().replace("-", "_")
    mtp_history_window_requested = getattr(args, "mtp_history_window", None)
    prefill_cache_cleanup_requested = bool(getattr(args, "prefill_cache_cleanup", False))
    no_prefill_cache_cleanup_requested = bool(
        getattr(args, "no_prefill_cache_cleanup", False)
    )
    if prefill_cache_cleanup_requested and no_prefill_cache_cleanup_requested:
        raise ValueError(
            "--prefill-cache-cleanup and --no-prefill-cache-cleanup cannot be "
            "used together"
        )
    prefill_cache_cleanup_every_requested = getattr(
        args, "prefill_cache_cleanup_every", None
    )
    prefill_chunk_size_requested = getattr(args, "prefill_chunk_size", None)
    clear_cache_every_requested = getattr(args, "clear_cache_every", None)
    defer_verify_hidden_requested = bool(
        getattr(args, "defer_verify_hidden_eval", False)
    )
    no_defer_verify_hidden_requested = bool(
        getattr(args, "no_defer_verify_hidden_eval", False)
    )
    if defer_verify_hidden_requested and no_defer_verify_hidden_requested:
        raise ValueError(
            "--defer-verify-hidden-eval and --no-defer-verify-hidden-eval "
            "cannot be used together"
        )
    verify_hidden_mode_requested = (
        str(getattr(args, "verify_hidden_mode", "") or "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    no_batch_target_arrays_requested = bool(
        getattr(args, "no_batch_target_arrays", False)
    )
    prefill_stock_cache_only_requested = bool(
        getattr(args, "prefill_stock_cache_only", False)
    )
    if paged_attn_impl_requested:
        profile_env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] = paged_attn_impl_requested
    if mtp_history_policy_requested:
        profile_env["MTPLX_MTP_HISTORY_POLICY"] = mtp_history_policy_requested
    if mtp_history_window_requested is not None:
        mtp_history_window_value = int(mtp_history_window_requested)
        if mtp_history_window_value <= 0:
            raise ValueError("--mtp-history-window must be positive")
        profile_env["MTPLX_MTP_HISTORY_LAST_WINDOW"] = str(mtp_history_window_value)
    if prefill_cache_cleanup_requested:
        profile_env["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] = "1"
    if no_prefill_cache_cleanup_requested:
        profile_env["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] = "0"
    if prefill_cache_cleanup_every_requested is not None:
        cleanup_every_text = str(prefill_cache_cleanup_every_requested).strip().lower()
        if cleanup_every_text == "auto":
            profile_env["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] = "auto"
        else:
            prefill_cache_cleanup_every_value = int(cleanup_every_text)
            if prefill_cache_cleanup_every_value <= 0:
                raise ValueError("--prefill-cache-cleanup-every must be positive")
            profile_env["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] = str(
                prefill_cache_cleanup_every_value
            )
    if prefill_chunk_size_requested is not None:
        prefill_chunk_size_value = int(prefill_chunk_size_requested)
        if prefill_chunk_size_value <= 0:
            raise ValueError("--prefill-chunk-size must be positive")
        profile_env["MTPLX_PREFILL_CHUNK_SIZE"] = str(prefill_chunk_size_value)
    if clear_cache_every_requested is not None:
        clear_cache_every_value = int(clear_cache_every_requested)
        if clear_cache_every_value < 0:
            raise ValueError("--clear-cache-every must be non-negative")
        profile_env["MTPLX_CLEAR_CACHE_EVERY"] = str(clear_cache_every_value)
    if defer_verify_hidden_requested:
        profile_env["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] = "1"
    elif no_defer_verify_hidden_requested:
        profile_env["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] = "0"
    if verify_hidden_mode_requested:
        profile_env["MTPLX_VERIFY_HIDDEN_MODE"] = verify_hidden_mode_requested
    if no_batch_target_arrays_requested:
        profile_env["MTPLX_BATCH_TARGET_ARRAYS"] = "0"
    if prefill_stock_cache_only_requested and _unsafe_stock_cache_only_allowed():
        profile_env["MTPLX_PREFILL_STOCK_CACHE_ONLY"] = "1"
    payload: dict[str, Any] = {
        "kind": "prefill_ladder",
        "git_sha": _git_sha(),
        "model": model,
        "profile": profile.to_dict(),
        "generation_mode": getattr(args, "generation_mode", None) or "mtp",
        "max_tokens": int(getattr(args, "max_tokens", 128)),
        "seed": int(getattr(args, "seed", None) or 0),
        "vary_seed_by_context": bool(getattr(args, "vary_seed_by_context", False)),
        "contexts": contexts,
        "inter_context_cache_cleanup": {
            "enabled": not bool(getattr(args, "no_inter_context_cache_cleanup", False)),
            "events": 0,
            "time_s": 0.0,
            "method": "mx.synchronize_default_then_clear_cache",
        },
        "hardware": inspect_hardware(),
        "env": profile_env,
        "prefill_layout": {
            "requested": prefill_layout,
            "env_value": prefill_layout_env_value,
        },
        "prompt": {
            "style": prompt_style,
            "format": prompt_format,
            "enable_thinking": enable_thinking,
            "policy": (
                "legacy_repeat_hard_truncate"
                if prompt_style == LEGACY_PROMPT_STYLE
                else PROMPT_POLICY_VERSION
            ),
            "tail_sha256": _sha256(prompt_tail) if prompt_style != LEGACY_PROMPT_STYLE else "",
            "tail_preview": (
                prompt_tail.strip().replace("\n", " ")[:240]
                if prompt_style != LEGACY_PROMPT_STYLE
                else ""
            ),
            "tail_preserved_by_default": prompt_style != LEGACY_PROMPT_STYLE,
            "release_valid": release_valid_prompt,
            "release_valid_reason": (
                "coherent final coding-agent request is preserved"
                if release_valid_prompt
                else "legacy or empty prompt tail is diagnostic-only"
            ),
        },
        "recommended_plugged_in_commands": _recommended_prefill_qa_commands(
            model=model,
            profile=profile.name,
            prompt_style=prompt_style,
            prompt_format=prompt_format,
            prefill_layout=prefill_layout,
            max_tokens=int(getattr(args, "max_tokens", 128)),
        ),
        "rows": [],
        "dry_run": bool(getattr(args, "dry_run", False)),
    }
    if paged_attn_impl_requested:
        payload["paged_attn_impl_override"] = paged_attn_impl_requested
    if mtp_history_policy_requested:
        payload["mtp_history_policy_override"] = mtp_history_policy_requested
    if mtp_history_window_requested is not None:
        payload["mtp_history_window_override"] = int(mtp_history_window_requested)
    if prefill_cache_cleanup_requested:
        payload["prefill_cache_cleanup_override"] = True
    if prefill_cache_cleanup_every_requested is not None:
        payload["prefill_cache_cleanup_every_override"] = (
            str(prefill_cache_cleanup_every_requested).strip().lower()
        )
    if prefill_chunk_size_requested is not None:
        payload["prefill_chunk_size_override"] = int(prefill_chunk_size_requested)
    if defer_verify_hidden_requested:
        payload["defer_verify_hidden_eval_override"] = True
    elif no_defer_verify_hidden_requested:
        payload["defer_verify_hidden_eval_override"] = False
    if verify_hidden_mode_requested:
        payload["verify_hidden_mode_override"] = verify_hidden_mode_requested
    if no_batch_target_arrays_requested:
        payload["batch_target_arrays_override"] = False
    if prefill_stock_cache_only_requested:
        payload["prefill_stock_cache_only_override"] = True
        if not _unsafe_stock_cache_only_allowed():
            payload["prefill_stock_cache_only_blocked"] = (
                f"requires {UNSAFE_STOCK_CACHE_ONLY_ALLOW_ENV}=1"
            )
    if payload["dry_run"]:
        return payload

    server_url = _ladder_server_url(args)
    if server_url is not None:
        # Everything below this point configures *this* process to host a model. In
        # serve mode the server under test owns that configuration, so the profile
        # env, the family lane, the allocator caps and the MX cache cleanup must not
        # be applied — or reproduced — here.
        return _run_prefill_ladder_against_server(
            args,
            payload=payload,
            base_url=server_url,
            contexts=contexts,
            prompt_style=prompt_style,
            prompt_format=prompt_format,
            prompt_tail=prompt_tail,
            enable_thinking=enable_thinking,
        )

    apply_profile_env(profile.name)
    family_verify_lane = _apply_family_verify_lane_override(model)
    ladder_verify_strategy, ladder_verify_core = _ladder_verify_route(model)
    payload["verify_route"] = {
        "verify_strategy": ladder_verify_strategy,
        "verify_core": ladder_verify_core,
    }
    if family_verify_lane:
        payload["family_verify_lane"] = {QWEN4_ENV_KEY: family_verify_lane}
    _apply_prefill_layout_override(prefill_layout)
    paged_attn_impl = _apply_paged_attention_impl_override(args)
    mtp_history_policy = _apply_mtp_history_policy_override(args)
    mtp_history_window = _apply_mtp_history_window_override(args)
    prefill_cache_cleanup = _apply_prefill_cache_cleanup_override(args)
    prefill_cache_cleanup_every = _apply_prefill_cache_cleanup_every_override(args)
    prefill_chunk_size = _apply_prefill_chunk_size_override(args)
    clear_cache_every = _apply_clear_cache_every_override(args)
    defer_verify_hidden = _apply_defer_verify_hidden_override(args)
    verify_hidden_mode = _apply_verify_hidden_mode_override(args)
    batch_target_arrays = _apply_batch_target_arrays_override(args)
    prefill_stock_cache_only = _apply_prefill_stock_cache_only_override(args)
    payload["env"] = _env_snapshot()
    if paged_attn_impl and paged_attn_impl != paged_attn_impl_requested:
        payload["paged_attn_impl_override"] = paged_attn_impl
    if mtp_history_policy and mtp_history_policy != mtp_history_policy_requested:
        payload["mtp_history_policy_override"] = mtp_history_policy
    if mtp_history_window is not None:
        payload["mtp_history_window_override"] = mtp_history_window
    if no_prefill_cache_cleanup_requested:
        payload["prefill_cache_cleanup_override"] = False
    elif prefill_cache_cleanup and not prefill_cache_cleanup_requested:
        payload["prefill_cache_cleanup_override"] = True
    if prefill_cache_cleanup_every is not None:
        payload["prefill_cache_cleanup_every_override"] = prefill_cache_cleanup_every
    if prefill_chunk_size is not None:
        payload["prefill_chunk_size_override"] = prefill_chunk_size
    if clear_cache_every is not None:
        payload["clear_cache_every_override"] = clear_cache_every
    if defer_verify_hidden is not None:
        payload["defer_verify_hidden_eval_override"] = defer_verify_hidden == "1"
    if verify_hidden_mode and verify_hidden_mode != verify_hidden_mode_requested:
        payload["verify_hidden_mode_override"] = verify_hidden_mode
    if batch_target_arrays is not None:
        payload["batch_target_arrays_override"] = batch_target_arrays == "1"
    if prefill_stock_cache_only and not prefill_stock_cache_only_requested:
        payload["prefill_stock_cache_only_override"] = True

    from .generation import generate_ar, generate_mtpk
    from .runtime import load
    from .sampling import SamplerConfig

    # Serve-path memory discipline (#261, F7): pin the exact Metal allocator
    # caps the serve path applies at startup BEFORE the model loads, and
    # refuse contexts beyond the model's context window instead of silently
    # benchmarking past the trained window. Uncapped ladder rows produced the
    # 102.6GB-at-262k class of headline the serve path can never reach.
    from mtplx.server.openai import apply_memory_caps_preflight

    payload["memory_preflight"] = apply_memory_caps_preflight(
        entry="bench.prefill_ladder",
        model=model,
        contexts=contexts,
    )

    max_session = None
    if getattr(args, "fanmax", False):
        from .thermal import MaxSession

        max_session = MaxSession(log=lambda line: print(line, flush=True))
        if not max_session.start():
            max_session = None

    try:
        rt = load(getattr(args, "model"), mtp=True)
        sampler = SamplerConfig(
            temperature=float(getattr(args, "temperature", 0.6)),
            top_p=float(getattr(args, "top_p", 0.95)),
            top_k=int(getattr(args, "top_k", 20)),
        )
        draft_sampler = SamplerConfig(
            temperature=float(getattr(args, "draft_temperature", None) or sampler.temperature),
            top_p=float(getattr(args, "draft_top_p", None) or sampler.top_p),
            top_k=int(getattr(args, "draft_top_k", None) or sampler.top_k),
        )
        generation_mode = getattr(args, "generation_mode", None) or "mtp"
        depth = int(getattr(args, "speculative_depth", 0) or 3)
        seed_base = int(getattr(args, "seed", None) or 0)
        vary_seed_by_context = bool(getattr(args, "vary_seed_by_context", False))
        inter_context_cleanup_enabled = not bool(
            getattr(args, "no_inter_context_cache_cleanup", False)
        )
        for index, context_tokens in enumerate(contexts):
            try:
                import mlx.core as mx

                mx.reset_peak_memory()
            except Exception:
                pass
            prompt = _prompt_build_for_context(
                rt.tokenizer,
                int(context_tokens),
                prompt_style=prompt_style,
                prompt_tail=prompt_tail,
                prompt_format=prompt_format,
                enable_thinking=enable_thinking,
            )
            prompt_ids = prompt.token_ids
            first_token_s: float | None = None

            def record_first(_tokens: list[int]) -> None:
                nonlocal first_token_s
                if first_token_s is None:
                    first_token_s = time.perf_counter()

            request_started_s = time.perf_counter()
            row_seed = seed_base + index if vary_seed_by_context else seed_base
            if generation_mode == "ar":
                out = generate_ar(
                    rt,
                    prompt_ids,
                    max_tokens=int(getattr(args, "max_tokens", 128)),
                    sampler=sampler,
                    seed=row_seed,
                    stop_token_ids=set(),
                    token_callback=record_first,
                )
            else:
                out = generate_mtpk(
                    rt,
                    prompt_ids,
                    max_tokens=int(getattr(args, "max_tokens", 128)),
                    sampler=sampler,
                    draft_sampler=draft_sampler,
                    speculative_depth=depth,
                    seed=row_seed,
                    mtp_hidden_variant="post_norm",
                    mtp_cache_policy="persistent",
                    mtp_history_policy="committed",
                    verify_strategy=ladder_verify_strategy,
                    verify_core=ladder_verify_core,
                    stop_token_ids=set(),
                    token_callback=record_first,
                )
            row = _row_from_output(
                context_tokens=int(context_tokens),
                output=out,
                request_started_s=request_started_s,
                first_token_s=first_token_s,
            )
            row.update(prompt.metadata)
            row["requested_prefill_layout"] = prefill_layout
            row["seed"] = row_seed
            payload["rows"].append(row)
            if inter_context_cleanup_enabled:
                # Per-row flush (#F7): every row — including the last, largest
                # context — releases allocator pressure before the next
                # measurement or process exit. The old last-row exclusion left
                # the biggest context's pool resident with no receipt.
                cleanup_time = _sync_and_clear_cache_between_contexts()
                cleanup_meta = payload["inter_context_cache_cleanup"]
                cleanup_meta["events"] = int(cleanup_meta["events"]) + 1
                cleanup_meta["time_s"] = float(cleanup_meta["time_s"]) + cleanup_time
                row["post_row_inter_context_cache_cleanup_time_s"] = cleanup_time
    finally:
        if max_session is not None:
            max_session.stop()

    return payload


def write_prefill_ladder(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def emit_prefill_ladder(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    _print_table(list(payload.get("rows") or []))
