"""The prefill ladder's serve harness: measure a live server, never load the pack here.

The in-process ladder cannot run some families — enabling a pack's own verify lane needs
the server's cache containers, so `runtime.load()` dies at
`unsupported_container:ArraysCache` on qwen4_exp. These tests pin the alternative: with
`--harness direct-http` the ladder keeps its prompt policy, delivers it over HTTP, and
must not touch the model loader, the profile environment, or the MX allocator of the
measuring process.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from argparse import Namespace
from types import SimpleNamespace

from mtplx import prefill_bench, runtime
from mtplx.prefill_bench import (
    _ladder_served_target,
    _ladder_server_row,
    _ladder_server_url,
    run_prefill_ladder,
)


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token)) for token in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = ""
        for message in messages:
            text += f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n"
        if add_generation_prompt:
            text += "<assistant>\n"
        return self.encode(text) if tokenize else text


def test_v1_is_added_once():
    assert prefill_bench._ladder_v1("http://h:9002") == "http://h:9002/v1"
    assert prefill_bench._ladder_v1("http://h:9002/v1/") == "http://h:9002/v1"


def _args(**overrides):
    base = {
        "contexts": "512,1k",
        "full": False,
        "profile": "sustained",
        "model": "fake-model",
        "generation_mode": "mtp",
        "max_tokens": 2,
        "dry_run": False,
        "prompt_style": "coding-agent",
        "prompt_format": "chat",
        "prefill_layout": "profile",
        "prompt_tail": "\n\n# Final user request\nPatch the harness.\n",
        "prompt_tail_file": None,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "draft_temperature": None,
        "draft_top_p": None,
        "draft_top_k": None,
        "speculative_depth": 3,
        "seed": 0,
        "fanmax": False,
        "disable_thinking": True,
        "enable_thinking": False,
        "harness": "auto",
        "url": "http://127.0.0.1:8000",
        "port": 8041,
    }
    base.update(overrides)
    return Namespace(**base)


class _Stream:
    """A streamed chat completion: deltas first, then the final usage chunk.

    Lines are bytes because that is what an HTTP response iterates over — the
    production reader decodes them.
    """

    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode("utf-8") for line in lines]

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Body:
    """A non-streamed JSON response (`/v1/models` reads the whole body)."""

    def __init__(self, payload: object) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._payload


def _sse(*, prompt_tokens: int, cached: int, completion: int) -> list[str]:
    usage_chunk = json.dumps(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
    )
    return [
        'data: {"choices":[{"delta":{"reasoning_content":"thin"}}]}\n',
        'data: {"choices":[{"delta":{"content":"O"}}]}\n',
        f"data: {usage_chunk}\n",
        "data: [DONE]\n\n",
    ]


# --- target resolution ---------------------------------------------------------


def test_harness_auto_stays_in_process():
    assert _ladder_server_url(_args()) is None


def test_default_url_is_mtplx_default_server_port():
    assert _ladder_server_url(_args(harness="direct-http")) == "http://127.0.0.1:8000"


def test_explicit_port_wins_over_the_default_url():
    assert _ladder_server_url(_args(harness="direct-http", port=9002)) == "http://127.0.0.1:9002"


def test_explicit_url_wins_and_loses_trailing_slash():
    url = _ladder_server_url(
        _args(harness="direct-http", url="https://box.example:9002/v1/")
    )
    assert url == "https://box.example:9002/v1"


def test_default_port_alone_does_not_redirect():
    # `--port` ships with a bench default; only a moved port means "that server".
    assert (
        _ladder_server_url(_args(harness="direct-http", port=8041, url="http://h:1"))
        == "http://h:1"
    )


# --- row arithmetic ------------------------------------------------------------


def test_context_beyond_the_servers_window_is_refused_before_sending(monkeypatch):
    """Same discipline as the in-process preflight: refuse, do not benchmark past it."""

    monkeypatch.setattr(
        prefill_bench,
        "_ladder_served_target",
        lambda base_url, **kwargs: {"model_id": "m", "context_length": 131072},
    )
    monkeypatch.setattr(
        runtime, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load"))
    )
    monkeypatch.setattr(
        prefill_bench,
        "_ladder_tokenizer",
        lambda model: (_ for _ in ()).throw(
            AssertionError("an over-window context must be refused before tokenizing")
        ),
    )
    try:
        run_prefill_ladder(_args(harness="direct-http", port=9002, contexts="131072", max_tokens=8))
    except RuntimeError as exc:
        assert "131072" in str(exc) and "advertises" in str(exc)
    else:
        raise AssertionError("an over-window context must be refused")


def test_server_row_separates_reuse_from_reprefill(monkeypatch):
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        seen["body"] = json.loads(request.data.decode())
        seen["url"] = request.full_url
        return _Stream(_sse(prompt_tokens=100_000, cached=4_000, completion=3))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    row = _ladder_server_row(
        "http://127.0.0.1:9002",
        context_tokens=100_000,
        text="x" * 10,
        model_id="mtplx-flash-next",
        max_tokens=8,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        seed=7,
        timeout_s=60.0,
    )

    assert seen["url"] == "http://127.0.0.1:9002/v1/chat/completions"
    assert seen["body"]["stream"] is True
    assert seen["body"]["model"] == "mtplx-flash-next"
    assert row["prompt_tokens"] == 100_000
    assert row["cached_tokens"] == 4_000
    assert row["new_prefill_tokens"] == 96_000
    assert row["generated_tokens"] == 3
    assert row["measure"] == "serve"
    assert row["ttft_s"] is not None and row["ttft_s"] > 0
    assert row["prompt_tps"] == row["new_prefill_tokens"] / row["ttft_s"]
    assert row["pp_tps"] == row["prompt_tps"]


def test_server_row_counts_a_mismatched_retokenisation(monkeypatch):
    """The server re-tokenizes text, so the requested and achieved sizes must both show."""

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: _Stream(
            _sse(prompt_tokens=1234, cached=0, completion=1)
        ),
    )
    row = _ladder_server_row(
        "http://h",
        context_tokens=1024,
        text="x",
        model_id="m",
        max_tokens=2,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=0,
        timeout_s=5.0,
    )

    assert row["context_tokens"] == 1024
    assert row["prompt_tokens"] == 1234
    assert row["context_delta_tokens"] == 210


def test_served_model_id_is_tolerant_of_a_dead_target(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert _ladder_served_target("http://127.0.0.1:1", timeout_s=1.0) == {}

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: _Body(
            {"data": [{"id": "mtplx-flash-next", "context_length": 131072}]}
        ),
    )
    assert _ladder_served_target("http://h", timeout_s=1.0) == {
        "model_id": "mtplx-flash-next",
        "context_length": 131072,
    }


# --- the ladder itself ---------------------------------------------------------


def test_serve_harness_never_loads_the_pack_or_touches_process_env(monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("serve harness must not load the model in-process")

    def fake_apply_profile_env(name, **kwargs):
        raise AssertionError("serve harness must not rewrite this process's profile env")

    monkeypatch.setattr(runtime, "load", forbidden_load)
    monkeypatch.setattr(prefill_bench, "apply_profile_env", fake_apply_profile_env)
    monkeypatch.setattr(prefill_bench, "_ladder_tokenizer", lambda model: _CharTokenizer())
    monkeypatch.setattr(
        prefill_bench,
        "_ladder_served_target",
        lambda base_url, **kwargs: {"model_id": "mtplx-flash-next", "context_length": 262144},
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: _Stream(
            _sse(prompt_tokens=640, cached=8, completion=2)
        ),
    )
    before_env = dict(os.environ)
    try:
        payload = run_prefill_ladder(_args(harness="direct-http", port=9002))
    finally:
        os.environ.clear()
        os.environ.update(before_env)

    assert payload["harness"] == "direct-http"
    assert payload["in_process_model_load"] is False
    assert payload["server"] == {
        "url": "http://127.0.0.1:9002",
        "model_id": "mtplx-flash-next",
        "context_length": 262144,
    }
    assert payload["inter_context_cache_cleanup"]["events"] == 0
    assert payload["recommended_plugged_in_commands"] == []
    assert "serve_harness_note" in payload
    assert len(payload["rows"]) == 2
    row = payload["rows"][0]
    assert row["new_prefill_tokens"] == 632
    assert row["prompt_policy"]  # the in-process prompt metadata is still recorded
    assert row["requested_prefill_layout"] == "profile"
    # No profile env was applied, so the process is exactly as it was found.
    assert os.environ == before_env


def test_short_generations_report_no_decode_rate(monkeypatch):
    """4 generated tokens cannot measure a decode rate; the row must say so."""

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: _Stream(_sse(prompt_tokens=512, cached=0, completion=4)),
    )
    row = _ladder_server_row(
        "http://h",
        context_tokens=512,
        text="x",
        model_id="m",
        max_tokens=4,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=0,
        timeout_s=5.0,
    )

    assert row["decode_tok_s"] is None
    assert row["prompt_tps"] > 0  # prefill is still measured: that is the ladder's job


def test_long_enough_generations_do_report_a_decode_rate(monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: _Stream(_sse(prompt_tokens=512, cached=0, completion=64)),
    )
    row = _ladder_server_row(
        "http://h",
        context_tokens=512,
        text="x",
        model_id="m",
        max_tokens=64,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        seed=0,
        timeout_s=5.0,
    )

    assert row["decode_tok_s"] is not None


def test_dead_server_names_the_target_not_a_traceback(monkeypatch):
    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    try:
        _ladder_server_row(
            "http://127.0.0.1:9999",
            context_tokens=512,
            text="x",
            model_id="m",
            max_tokens=2,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=0,
            timeout_s=1.0,
        )
    except RuntimeError as exc:
        assert "http://127.0.0.1:9999" in str(exc)
        assert "mtplx serve" in str(exc)
    else:
        raise AssertionError("an unreachable server must raise, not return a row")


def test_in_process_harness_still_loads_and_reports_no_server_key(monkeypatch):
    loaded = {"count": 0}

    def fake_load(model, *, mtp):
        loaded["count"] += 1
        return SimpleNamespace(tokenizer=_CharTokenizer())

    monkeypatch.setattr(runtime, "load", fake_load)
    monkeypatch.setattr(
        prefill_bench, "_row_from_output", lambda **kwargs: {"context_tokens": 0}
    )
    monkeypatch.setattr(
        "mtplx.generation.generate_mtpk",
        lambda rt, ids, **kwargs: SimpleNamespace(tokens=[1], stats={}),
    )
    import mtplx.server.openai as openai_server

    monkeypatch.setattr(
        openai_server, "apply_memory_caps_preflight", lambda **kwargs: {"stub": True}
    )
    before_env = dict(os.environ)
    try:
        payload = run_prefill_ladder(_args(contexts="512"))
    finally:
        os.environ.clear()
        os.environ.update(before_env)

    assert loaded["count"] == 1
    assert "harness" not in payload or payload.get("harness") != "direct-http"
    assert "server" not in payload
