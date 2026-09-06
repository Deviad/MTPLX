"""Task 3 step 2 of the 2026-09-06 handoff plan: the unit seam behind
``paged_gqa_sdpa_calls = 0`` on every served qwen4_exp row.

Step 1's instrumentation proved no counted lane fires under either sustained
layout; these tests pin WHY at unit level, so "unreachable" is a proven
property of the code instead of an inference from absent counters. Four gates
keep the lane at zero on the serve path, in the order a request meets them:

1. layout scope -- ``_target_prefill_cache_layout_scope`` force-zeroes every
   owned-attention env while either sustained layout is active, so
   ``_make_target_prefill_cache`` can never install the subsystem.
2. wiring -- ``install_vllm_metal_paged_attention_kv_cache`` converts only
   entries exposing stock ``keys``/``values``; qwen4_exp's full-attention
   layers carry ``QSACache`` (the KV lives in ``.kv``, beside positional
   indexer streams), so the install skips them even with every env on and
   the inner KV already materialized.
3. impl gate -- the GQA route block inside ``paged_attention`` runs only
   under ``MTPLX_VLLM_METAL_PAGED_ATTN_IMPL`` in {sdpa_2pass_paged,
   mlx_vector_paged} with offset past the 1024 two-pass threshold; the serve
   default never sets it.
4. route gate -- the route is off by default and its default q window
   (min_q 4, max_q 5) is decode/MTP width: an 8192-token prefill chunk is
   refused with ``q_len_gt_max`` even with the route on.

The last three tests drive the real entry points and prove the positive
half: wired + paged impl + decode-width q increments ``gqa_sdpa_calls`` with
route and phase recorded, a prefill-width chunk increments
``partitioned_paged_calls`` per phase, and without a paged impl the same
call is served with no lane counter and no miss -- the one place "runs
uncounted" is real. Zero on serve rows therefore means "never wired", not
"kernels ran uncounted".

Head shapes come from the Flash-Next pack config (24 query heads, 2 KV
heads, head_dim 256, full_attention_interval 4 -> 12 QSA layers of 48).
"""

import os

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mtplx import attention_context
from mtplx.cache_state import (
    VllmMetalPagedKVCache,
    _paged_gqa_sdpa_route_decision_from_env,
    install_vllm_metal_paged_attention_kv_cache,
)
from mtplx.generation import (
    _contiguous_prefill_cache_layout_enabled,
    _make_target_prefill_cache,
    _sustained_prefill_layout,
    _target_prefill_cache_layout_scope,
)
from mtplx.models.qwen4_exp import QSACache

QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
# The default route window is decode/MTP width: 4..5 tokens per step.
DECODE_WIDTH = 4
# The sustained prefill chunk size the serve wrappers run.
PREFILL_CHUNK = 8192

OWNED_ENV_KEYS = (
    "MTPLX_VLLM_METAL_PAGED_ATTN",
    "MTPLX_OWNED_ATTN_KV",
    "MTPLX_BLOCK_OWNED_ATTN_KV",
)

ROUTE_ENVS = (
    "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE",
    "MTPLX_PAGED_GQA_SDPA_ROUTE",
    "MTPLX_VLLM_METAL_PAGED_GQA_SDPA",
    "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT",
    "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_Q",
    "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MAX_Q",
    "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL",
    "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD",
    "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE",
    "MTPLX_VLLM_METAL_PAGED_ATTN_MAX_Q",
    "MTPLX_VLLM_METAL_PAGED_ATTN_2PASS_THRESHOLD",
    "MTPLX_SUSTAINED_PREFILL_LAYOUT",
    "MTPLX_SUSTAINED_PREFILL",
)

def _clear_lane_envs(monkeypatch):
    for key in ROUTE_ENVS + OWNED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _stock_kv_cache(tokens: int = 2048) -> KVCache:
    kv = KVCache()
    keys = mx.random.uniform(shape=(1, KV_HEADS, tokens, HEAD_DIM)).astype(
        mx.bfloat16
    )
    values = mx.random.uniform(shape=(1, KV_HEADS, tokens, HEAD_DIM)).astype(
        mx.bfloat16
    )
    kv.update_and_fetch(keys, values)
    return kv


def _queries(q_len: int) -> mx.array:
    return mx.random.uniform(shape=(1, QUERY_HEADS, q_len, HEAD_DIM)).astype(
        mx.bfloat16
    )


def _in_prefill_phase():
    return attention_context._ATTENTION_PHASE.set("prefill")


# --- gate 4: the route decision, as a pure function of env and shape -------


def test_route_is_off_by_default(monkeypatch):
    _clear_lane_envs(monkeypatch)
    decision = _paged_gqa_sdpa_route_decision_from_env(
        q_len=DECODE_WIDTH,
        offset=65536 + DECODE_WIDTH,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
    )
    assert decision.route == ""
    assert decision.reason == "disabled"


def test_route_accepts_the_pack_shape_at_decode_width(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "auto")
    decision = _paged_gqa_sdpa_route_decision_from_env(
        q_len=DECODE_WIDTH,
        offset=65536 + DECODE_WIDTH,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
    )
    assert decision.route == "async_per_head"
    assert decision.reason == "enabled"


def test_route_refuses_prefill_chunk_width_even_enabled(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "auto")
    # A prefill chunk past the 65536 min-context: refused for its width, so
    # no sustained-prefill sweep can ever move this counter by tuning.
    decision = _paged_gqa_sdpa_route_decision_from_env(
        q_len=PREFILL_CHUNK,
        offset=103_714,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
    )
    assert decision.route == ""
    assert decision.reason == "q_len_gt_max"
    # Decode width but still inside the early context: also refused.
    early = _paged_gqa_sdpa_route_decision_from_env(
        q_len=DECODE_WIDTH,
        offset=8192,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
    )
    assert early.reason == "context_lt_min"


def test_route_refuses_non_gqa_heads(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "auto")
    decision = _paged_gqa_sdpa_route_decision_from_env(
        q_len=DECODE_WIDTH,
        offset=65536 + DECODE_WIDTH,
        query_heads=24,
        kv_heads=24,
    )
    assert decision.route == ""
    assert decision.reason == "not_gqa"


# --- gate 1: the sustained layout scope ------------------------------------


@pytest.mark.parametrize(
    "layout", ["contiguous_dense_decode", "contiguous_then_repage"]
)
def test_sustained_layout_scope_zeroes_owned_attention_envs(
    monkeypatch, layout
):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", layout)
    for key in OWNED_ENV_KEYS:
        monkeypatch.setenv(key, "1")
    with _target_prefill_cache_layout_scope():
        for key in OWNED_ENV_KEYS:
            assert os.environ[key] == "0", key
    for key in OWNED_ENV_KEYS:
        assert os.environ[key] == "1", key


def test_layout_auto_resolves_into_the_scope(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    assert _sustained_prefill_layout() in {
        "contiguous_dense_decode",
        "contiguous_then_repage",
    }
    assert _contiguous_prefill_cache_layout_enabled()


def test_no_layout_means_no_scope(monkeypatch):
    _clear_lane_envs(monkeypatch)
    assert _sustained_prefill_layout() == ""
    assert not _contiguous_prefill_cache_layout_enabled()


# --- gate 2: the install cannot wire this family ----------------------------


def _written_qsacache() -> QSACache:
    cache = QSACache(4)
    keys = mx.random.uniform(shape=(1, KV_HEADS, 64, HEAD_DIM)).astype(
        mx.bfloat16
    )
    values = mx.random.uniform(shape=(1, KV_HEADS, 64, HEAD_DIM)).astype(
        mx.bfloat16
    )
    cache.kv.update_and_fetch(keys, values)
    return cache


def test_paged_install_skips_qwen4_exp_qsa_cache(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    cache = _written_qsacache()
    stats = install_vllm_metal_paged_attention_kv_cache(
        [cache], block_size=16, num_blocks=64
    )
    # QSACache hides the KV inside .kv: no stock keys/values to convert, so
    # the family's full-attention layers can never become paged caches.
    assert stats["entries"] == 0
    assert stats["skipped"] == 1
    assert isinstance(cache, QSACache)
    assert not isinstance(cache, VllmMetalPagedKVCache)


class _FakeQwen4Runtime:
    """The make_cache contract of MTPLXRuntime for a qwen4_exp layer set."""

    def make_cache(self):
        from mtplx.cache_state import configure_tail_owned_attention_kv_cache

        cache = [_written_qsacache()]
        configure_tail_owned_attention_kv_cache(cache)
        return cache


def test_target_prefill_cache_never_carries_owned_lanes(monkeypatch):
    _clear_lane_envs(monkeypatch)
    # Every owned-attention env ON, sustained layout active: the scope
    # zeroes them during construction, so nothing is installed.
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "contiguous_then_repage")
    cache = _make_target_prefill_cache(_FakeQwen4Runtime())
    assert len(cache) == 1
    assert isinstance(cache[0], QSACache)
    assert not isinstance(cache[0], VllmMetalPagedKVCache)


# --- the positive half: the lanes count when the object is in the path -----


def test_gqa_lane_counts_when_wired_at_decode_width(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "auto")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT", "8")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_ATTN_IMPL", "mlx_vector_paged")
    paged = VllmMetalPagedKVCache.from_cache(
        _stock_kv_cache(tokens=2048), block_size=16, num_blocks=256
    )
    token = _in_prefill_phase()
    try:
        out = paged.paged_attention(
            _queries(DECODE_WIDTH), scale=0.0625, sliding_window=-1, mask=None
        )
    finally:
        attention_context._ATTENTION_PHASE.reset(token)
    assert tuple(out.shape) == (1, QUERY_HEADS, DECODE_WIDTH, HEAD_DIM)
    assert paged.gqa_sdpa_calls == 1
    assert paged.gqa_sdpa_calls_by_route == {"async_per_head": 1}
    assert paged.gqa_sdpa_calls_by_phase == {"prefill": 1}


def test_gqa_route_block_is_silent_without_paged_impl(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "auto")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_MIN_CONTEXT", "8")
    paged = VllmMetalPagedKVCache.from_cache(
        _stock_kv_cache(tokens=2048), block_size=16, num_blocks=256
    )
    token = _in_prefill_phase()
    try:
        out = paged.paged_attention(
            _queries(DECODE_WIDTH), scale=0.0625, sliding_window=-1, mask=None
        )
    finally:
        attention_context._ATTENTION_PHASE.reset(token)
    # Served correctly, but by a lane with no counter and no recorded miss:
    # the impl gate keeps the route block from ever running. This is the one
    # place "runs uncounted" is real -- inside the owned object.
    assert tuple(out.shape) == (1, QUERY_HEADS, DECODE_WIDTH, HEAD_DIM)
    assert paged.gqa_sdpa_calls == 0
    assert paged.gqa_sdpa_route_misses_by_phase_reason == {}
    assert paged.paged_attention_calls == 1


def test_partitioned_paged_counts_prefill_width(monkeypatch):
    _clear_lane_envs(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN", "1")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "1024")
    paged = VllmMetalPagedKVCache.from_cache(
        _stock_kv_cache(tokens=2048), block_size=16, num_blocks=256
    )
    token = _in_prefill_phase()
    try:
        out = paged.paged_attention(
            _queries(PREFILL_CHUNK), scale=0.0625, sliding_window=-1, mask=None
        )
    finally:
        attention_context._ATTENTION_PHASE.reset(token)
    # The prefill-width owned lane: a real paged kernel, counted per phase.
    assert tuple(out.shape) == (1, QUERY_HEADS, PREFILL_CHUNK, HEAD_DIM)
    assert paged.partitioned_paged_calls == 1
    assert paged.partitioned_paged_calls_by_phase == {"prefill": 1}
