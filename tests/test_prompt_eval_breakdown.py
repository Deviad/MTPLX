"""The prompt phase reports its parts, not only its total.

A 104 k follow-up cost 2.835 s for a 675-token suffix (4.19 ms per new token,
against 2.59 ms per token for the *cold* prefill of the same context) and the
rows could not say what the time was: `prompt_eval_time_s` is composed at four
different places from up to four measured parts -- a one-token repair forward
against the restored KV, the chunked suffix prefill, the MTP-history rebuild,
and repaging -- but only the total was emitted.

These tests pin the two halves of the fix. `_prompt_eval_time_from_parts` is the
one place that composes the aggregate, so a part cannot be summed into the total
without being emitted beside it. And `prompt_eval_breakdown_complete` keeps a
0.0 part from meaning two things: only the bank-restore paths decompose, so a
cold row (total = the whole prefill, every part 0.0) is marked incomplete rather
than reading as "no repair, no suffix, no repage". That is the same ambiguity
`prefill_attention_impl` had before it grew `"none"` versus `"unrecorded"`.
"""

import dataclasses

import pytest

from mtplx.generation import (
    GenerationStats,
    PromptState,
    _prompt_eval_breakdown_is_complete,
    _prompt_eval_time_from_parts,
)
from mtplx.server.openai import (
    PUBLIC_MTPLX_STATS_KEYS,
    REQUEST_ENVELOPE_PROMPT_BREAKDOWN_KEYS,
)

PART_FIELDS = (
    "prompt_mtp_history_time_s",
    "prompt_repair_time_s",
    "prompt_suffix_time_s",
    "prompt_repage_time_s",
    "prompt_eval_breakdown_complete",
)


def _state(**times) -> PromptState:
    return PromptState(
        trunk_cache=[],
        logits=None,
        hidden=None,
        committed_mtp_cache=None,
        token_prefix=(),
        **times,
    )


def test_parts_compose_the_total():
    total = _prompt_eval_time_from_parts(
        repair_s=0.119, suffix_s=2.516, mtp_history_s=0.2
    )
    assert total == pytest.approx(2.835)
    # Every part the four restore paths can produce reaches the sum.
    assert _prompt_eval_time_from_parts(repage_s=0.4) == pytest.approx(0.4)
    assert _prompt_eval_time_from_parts(
        repair_s=0.1, suffix_s=0.2, mtp_history_s=0.3, repage_s=0.4
    ) == pytest.approx(1.0)


def test_negative_parts_cannot_subtract_from_the_total():
    # A part measured as negative (clock or ordering bug) must not shrink the
    # aggregate below what was actually spent.
    assert _prompt_eval_time_from_parts(repair_s=-5.0, suffix_s=1.0) == pytest.approx(1.0)


def test_complete_only_when_the_parts_account_for_the_total():
    # The prefix-hit path: repair + suffix + mtp history.
    restored = _state(
        prompt_eval_time_s=_prompt_eval_time_from_parts(
            repair_s=0.119, suffix_s=2.516, mtp_history_s=0.2
        ),
        prompt_repair_time_s=0.119,
        prompt_suffix_time_s=2.516,
        prompt_mtp_history_time_s=0.2,
    )
    assert _prompt_eval_breakdown_is_complete(restored)

    # Full hit with no suffix: repair + repage.
    repaged = _state(
        prompt_eval_time_s=_prompt_eval_time_from_parts(repair_s=0.1, repage_s=0.3),
        prompt_repair_time_s=0.1,
        prompt_repage_time_s=0.3,
    )
    assert _prompt_eval_breakdown_is_complete(repaged)

    # A cold prompt: the total is the whole prefill and no part was measured.
    cold = _state(prompt_eval_time_s=268.64)
    assert not _prompt_eval_breakdown_is_complete(cold)

    # A part dropped from the sum must not pass as complete.
    dropped = _state(prompt_eval_time_s=2.835, prompt_suffix_time_s=2.516)
    assert not _prompt_eval_breakdown_is_complete(dropped)


def test_the_exact_zero_total_is_complete_not_merely_unmeasured():
    # A full cache hit with nothing to do: total 0.0 and every part 0.0 is a
    # measurement, so it reads complete.
    assert _prompt_eval_breakdown_is_complete(_state(prompt_eval_time_s=0.0))


def test_prompt_state_and_stats_carry_the_parts():
    state_fields = {f.name: f for f in dataclasses.fields(PromptState)}
    stats_fields = {f.name: f for f in dataclasses.fields(GenerationStats)}
    for name in ("prompt_repair_time_s", "prompt_suffix_time_s", "prompt_repage_time_s"):
        assert name in state_fields, name
        assert state_fields[name].default == 0.0, name
        assert name in stats_fields, name
        assert stats_fields[name].default == 0.0, name
    assert "prompt_eval_breakdown_complete" in stats_fields
    assert stats_fields["prompt_eval_breakdown_complete"].default is False


def test_envelope_keys_reach_both_projections():
    # A key that is not a GenerationStats field can never satisfy the
    # envelope's `if key in stats`, and would silently never be emitted.
    stats_fields = {f.name for f in dataclasses.fields(GenerationStats)}
    for key in REQUEST_ENVELOPE_PROMPT_BREAKDOWN_KEYS:
        assert key in stats_fields, f"{key} is not a GenerationStats field"
        assert key in PUBLIC_MTPLX_STATS_KEYS, f"{key} missing from the public projection"
    assert set(REQUEST_ENVELOPE_PROMPT_BREAKDOWN_KEYS) == set(PART_FIELDS)
    # The four measured parts plus the completeness flag must ride together: a
    # row carrying three of the four parts cannot be summed by whoever reads it,
    # which is the whole point of emitting the breakdown.
    assert len(REQUEST_ENVELOPE_PROMPT_BREAKDOWN_KEYS) == 5
