"""The parity tolerance is a declared policy, not a reaction to a red test.

`compare_verify_outputs` claims bit-exactness, and on real hardware that is true for
cache/state leaves and false for the readout: `mx.compile` fuses it in a different
floating-point order than eager. This suite's own maintainer docstring already says so
(3e-5 on a q8 toy, 0.385 on a plain paged toy, state bit-identical both times). What was
missing was the code agreeing with the documentation, which is what these tests pin:

  * state/capture leaves: always exact, tolerance never applies
  * logits/hidden: within ``atol + rtol*|reference|``, defaults 1e-5 / 2e-5, env-settable
  * shape, dtype and missing-leaf differences: never tolerated, at any tolerance
"""

import numpy as np
import pytest

from mtplx.graphbank import (
    COMPILED_VERIFY_PARITY_ATOL_ENV,
    COMPILED_VERIFY_PARITY_RTOL_ENV,
    DEFAULT_COMPILED_VERIFY_PARITY_ATOL,
    DEFAULT_COMPILED_VERIFY_PARITY_RTOL,
    CompiledVerifyBank,
    _OUTPUT_PARITY_NAMES,
    _parity_output_atol,
    _parity_output_rtol,
    compare_verify_outputs,
)

STATES = "state[0:gdn].0"


def leaves(**over):
    base = np.zeros((4,), dtype=np.float32)
    out = {"logits": base, "hidden": base, STATES: base}
    out.update(over)
    return out


def test_defaults_are_small_and_documented():
    assert DEFAULT_COMPILED_VERIFY_PARITY_ATOL == 1e-5
    assert DEFAULT_COMPILED_VERIFY_PARITY_RTOL == 2e-5
    assert _parity_output_atol() == DEFAULT_COMPILED_VERIFY_PARITY_ATOL
    assert _parity_output_rtol() == DEFAULT_COMPILED_VERIFY_PARITY_RTOL
    assert _OUTPUT_PARITY_NAMES == frozenset({"logits", "hidden"})


def test_relative_term_covers_a_large_scale_readout():
    # The shape the quantized toy actually has: |values| ~425, absolute drift 3.418e-03.
    # An absolute-only knob cannot express that, which is why rtol exists.
    ref = leaves(hidden=np.full((4,), 425.15, dtype=np.float32))
    cand = dict(ref, hidden=ref["hidden"] + np.float32(3.418e-03))
    assert compare_verify_outputs(ref, cand) == []


def test_relative_term_does_not_become_a_blanket_for_large_errors():
    ref = leaves(hidden=np.full((4,), 425.15, dtype=np.float32))
    cand = dict(ref, hidden=ref["hidden"] + np.float32(1e-2))
    joined = "\n".join(compare_verify_outputs(ref, cand))
    assert "hidden: value mismatch" in joined


def test_in_band_output_difference_is_not_a_mismatch():
    ref = leaves()
    cand = leaves(logits=ref["logits"] + 3e-6, hidden=ref["hidden"] + 3e-6)
    assert compare_verify_outputs(ref, cand) == []


def test_out_of_band_output_difference_is_still_reported():
    ref = leaves()
    cand = leaves(logits=ref["logits"] + 1e-3)
    report = "\n".join(compare_verify_outputs(ref, cand))
    assert "logits: value mismatch" in report


def test_state_leaves_are_never_tolerated(monkeypatch):
    # The whole point of the policy: widening the output tolerance must not be able to
    # hide a cache divergence, which is the failure mode compiled verify exists to catch.
    monkeypatch.setenv(COMPILED_VERIFY_PARITY_ATOL_ENV, "1e3")
    monkeypatch.setenv(COMPILED_VERIFY_PARITY_RTOL_ENV, "1e3")
    ref = leaves()
    cand = leaves(**{STATES: ref[STATES] + 1e-9})
    report = "\n".join(compare_verify_outputs(ref, cand))
    assert f"{STATES}: value mismatch" in report


def test_shape_dtype_and_missing_survive_a_large_atol(monkeypatch):
    monkeypatch.setenv(COMPILED_VERIFY_PARITY_ATOL_ENV, "1e3")
    monkeypatch.setenv(COMPILED_VERIFY_PARITY_RTOL_ENV, "1e3")
    ref = leaves()
    cand = {
        "logits": np.zeros((5,), dtype=np.float32),  # shape
        "hidden": np.zeros((4,), dtype=np.float16),  # dtype
        # STATES missing entirely
    }
    joined = "\n".join(compare_verify_outputs(ref, cand))
    assert "logits: shape mismatch" in joined
    assert "hidden: dtype mismatch" in joined
    assert f"{STATES}: missing from candidate output" in joined


def test_both_knobs_zero_restores_bit_exact_gate_a(monkeypatch):
    monkeypatch.setenv(COMPILED_VERIFY_PARITY_ATOL_ENV, "0")
    monkeypatch.setenv(COMPILED_VERIFY_PARITY_RTOL_ENV, "0")
    assert _parity_output_atol() == 0.0
    assert _parity_output_rtol() == 0.0
    ref = leaves()
    cand = leaves(logits=ref["logits"] + 1e-9)
    assert "logits: value mismatch" in "\n".join(compare_verify_outputs(ref, cand))
    big = leaves(hidden=np.full((4,), 425.15, dtype=np.float32))
    big_cand = dict(big, hidden=big["hidden"] + np.float32(3.418e-03))
    assert "hidden: value mismatch" in "\n".join(compare_verify_outputs(big, big_cand))


def test_bad_env_value_does_not_silently_disable_the_check(monkeypatch):
    monkeypatch.setenv(COMPILED_VERIFY_PARITY_ATOL_ENV, "on")
    assert _parity_output_atol() == DEFAULT_COMPILED_VERIFY_PARITY_ATOL


def test_bank_threads_the_tolerance_into_production_parity():
    # The knob must reach the production parity path, not just the standalone
    # comparator: with atol=0 the fp32 toy raises exactly like it did before this
    # change, and with the default tolerance the same session passes clean.
    import importlib.util as iu

    import mlx.core as mx

    from mtplx.graphbank import CompiledVerifyParityError

    spec = iu.spec_from_file_location(
        "tcv_for_tolerance", "tests/test_graphbank_compiled_verify.py"
    )
    tcv = iu.module_from_spec(spec)
    spec.loader.exec_module(tcv)

    def run(output_atol, output_rtol):
        rt = tcv.ToyHybridRuntime(seed=7)
        bank = CompiledVerifyBank(
            rt, parity=True, output_atol=output_atol, output_rtol=output_rtol
        )
        cache = tcv._prefill(rt, [0, 1, 2])
        for window in tcv.VERIFY_WINDOWS[:2]:
            bank.forward_ar_capture(mx.array([window]), cache=cache)
        return bank

    with pytest.raises(CompiledVerifyParityError):
        run(0.0, 0.0)

    tolerant = run(None, None)
    assert tolerant.stats["parity_checks"] == 2
    assert tolerant.stats["parity_failures"] == 0


def test_default_stays_below_the_divergence_the_suite_injects():
    """Why the default cannot be raised to cover the quantized toy.

    The parity2 tests prove detection by adding 1e-3 to logits; the quantized toy's
    natural readout drift is larger than that (3.418e-03 measured). One default could
    not both tolerate that drift and detect the injection, so the drift is handled by a
    declared `output_atol` at that one site and the default stays below 1e-3.
    """
    from mtplx.graphbank import _parity_output_atol, _parity_output_rtol

    atol, rtol = _parity_output_atol(), _parity_output_rtol()
    assert atol + rtol * 1.679 < 1e-3  # 1.679 = max |logits| of the fp32 toy
    assert atol + rtol * 3.843 < 1e-3  # 3.843 = max |hidden| of the fp32 toy
