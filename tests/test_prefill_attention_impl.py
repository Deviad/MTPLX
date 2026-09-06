"""The per-request row must name the prefill lane that actually ran.

Requested-versus-executed is what makes a sweep falsifiable. Before this, "the
paged lane was not chosen" was inferred from a key being absent, which is also
what a missing instrument looks like. These tests pin the derived field, the
phase discipline it depends on, and the two plumbing points that carry it: the
in-process bench row and the server's public engine-row allowlist.
"""

from dataclasses import fields
from types import SimpleNamespace

from mtplx.generation import (
    PREFILL_IMPL_NONE,
    GenerationStats,
    derive_prefill_attention_impl,
)
from mtplx.prefill_bench import _row_from_output


def _counters(
    *,
    partitioned: int = 0,
    gqa_by_phase: dict | None = None,
    dense: int = 0,
    large_q_split: int = 0,
):
    return SimpleNamespace(
        prefill_partitioned_paged_calls=partitioned,
        paged_gqa_sdpa_calls_by_phase=gqa_by_phase if gqa_by_phase is not None else {},
        prefill_dense_fallback_calls=dense,
        prefill_large_q_split_sdpa_fallback_calls=large_q_split,
    )


def test_no_lane_recorded_is_a_value_not_an_absence():
    assert derive_prefill_attention_impl(_counters()) == PREFILL_IMPL_NONE
    assert PREFILL_IMPL_NONE == "none"


def test_single_lane_carries_its_call_count():
    assert (
        derive_prefill_attention_impl(_counters(partitioned=3))
        == "partitioned_paged:3"
    )


def test_decode_phase_calls_are_not_prefill_evidence():
    # The one mistake that would make this field lie: counting a decode-phase
    # paged call as proof that prefill used the paged lane.
    stats = _counters(gqa_by_phase={"decode": 40, "postcommit": 7})
    assert derive_prefill_attention_impl(stats) == "none"
    assert (
        derive_prefill_attention_impl(_counters(gqa_by_phase={"prefill": 5, "decode": 40}))
        == "paged_gqa_sdpa:5"
    )


def test_every_lane_that_fired_is_named():
    stats = _counters(dense=2, large_q_split=1)
    assert (
        derive_prefill_attention_impl(stats)
        == "dense_fallback:2,large_q_split_sdpa_fallback:1"
    )


def test_unusable_counter_values_are_not_lane_evidence():
    assert derive_prefill_attention_impl(_counters(partitioned=None)) == "none"
    assert derive_prefill_attention_impl(_counters(dense="unrecorded")) == "none"
    assert derive_prefill_attention_impl(_counters(gqa_by_phase=None)) == "none"


def test_stats_defaults_to_unrecorded_not_none():
    names = {f.name for f in fields(GenerationStats)}
    assert {"prefill_attention_impl", "prefill_layout", "prefill_route"} <= names
    fresh = GenerationStats("mtp", 0, 0.0, 0.0)
    assert fresh.prefill_attention_impl == "unrecorded"
    assert fresh.prefill_layout == ""


def test_unrecorded_and_none_are_different_claims():
    # "none" = the lane accounting ran and no lane fired. "unrecorded" = the request
    # never reached that accounting. Collapsing them would let an uninstrumented
    # server look like evidence that no paged lane exists.
    assert PREFILL_IMPL_NONE == "none"
    assert derive_prefill_attention_impl(_counters()) == PREFILL_IMPL_NONE
    assert GenerationStats("mtp", 0, 0.0, 0.0).prefill_attention_impl != PREFILL_IMPL_NONE


def test_in_process_row_carries_impl_and_effective_layout():
    stats = {
        "prefill_attention_impl": "dense_fallback:4",
        "prefill_layout": "contiguous_paged_q8",
        "prefill_route": "contiguous_paged_q8",
        "generated_tokens": 1,
        "prompt_tps": 640.0,
    }
    output = SimpleNamespace(stats=stats, tokens=[1])
    row = _row_from_output(
        context_tokens=1024,
        output=output,
        request_started_s=0.0,
        first_token_s=1.0,
    )
    assert row["prefill_attention_impl"] == "dense_fallback:4"
    assert row["prefill_layout"] == "contiguous_paged_q8"


def test_absent_impl_key_reads_as_unrecorded_not_empty():
    stats = {"prefill_route": "x"}  # an older stats object, pre-derivation
    output = SimpleNamespace(stats=stats, tokens=[])
    row = _row_from_output(
        context_tokens=8,
        output=output,
        request_started_s=0.0,
        first_token_s=None,
    )
    assert row["prefill_attention_impl"] == "unrecorded"


def test_engine_row_allowlist_exposes_both_keys():
    # The serve harness cannot read these over HTTP; the sweep reads the daemon
    # row, so membership here is what makes a served sweep falsifiable.
    from mtplx.server.openai import PUBLIC_MTPLX_STATS_KEYS

    assert "prefill_attention_impl" in PUBLIC_MTPLX_STATS_KEYS
    assert "prefill_layout" in PUBLIC_MTPLX_STATS_KEYS
    assert "prefill_route" in PUBLIC_MTPLX_STATS_KEYS


def test_request_log_envelope_carries_the_lane_keys():
    """The keys have to reach the JSONL row, not just the stats object.

    That distinction is the whole history of this instrument: the 2026-09-06 16:36
    sweep read four lane counters as 0 and could not tell a measurement from a
    dataclass default, because the request-log envelope copied a different key set
    than the one carrying the layout.
    """
    from dataclasses import fields

    from mtplx.server.openai import REQUEST_ENVELOPE_LANE_KEYS

    assert {
        "prefill_route",
        "prefill_layout",
        "prefill_attention_impl",
    } <= set(REQUEST_ENVELOPE_LANE_KEYS)
    # An envelope key with no matching stats field is copied by `if key in stats`
    # and then silently never appears -- which is indistinguishable from "unmeasured".
    names = {f.name for f in fields(GenerationStats)}
    missing = sorted(set(REQUEST_ENVELOPE_LANE_KEYS) - names)
    assert not missing, f"envelope keys not present on GenerationStats: {missing}"
