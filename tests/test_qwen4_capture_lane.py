"""The generic GDN capture must not be entered by a trunk it was not written for.

``mtplx.gdn_capture.forward_with_gdn_capture`` walks ``inner.layers`` assuming the
qwen3_5/laguna layer vocabulary. ``qwen4_exp`` names its norms differently
(``q_layernorm``/``k_layernorm``/``hc_norm``) and gates on ``is_linear``, so when its
own verify lane is not installed the walk dies with
``AttributeError: 'DecoderLayer' object has no attribute 'input_layernorm'`` — which
reads as broken model code rather than a missing lane, and is what ``bench
prefill-ladder`` hit on a real Flash-Next pack (2.11.0, reconfirmed on 2.11.1).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.gdn_capture import generic_hybrid_capture_blocker
from mtplx.prefill_bench import _apply_family_verify_lane_override
from mtplx.runtime import MTPLXRuntime

QWEN4_VERIFY_ENV = "MTPLX_QWEN4_FIXED_M4_VERIFY"


def _model(*, with_qwen35_norms: bool):
    """A hybrid trunk: inner carries fa_idx/ssm_idx, so it is not the
    uniform-full-attention case the capture already routes around."""
    attrs = {"is_linear": True}
    if with_qwen35_norms:
        attrs["input_layernorm"] = lambda x: x
        attrs["post_attention_layernorm"] = lambda x: x
    inner = SimpleNamespace(layers=[SimpleNamespace(**attrs)], fa_idx=[1], ssm_idx=[0])
    return SimpleNamespace(language_model=SimpleNamespace(model=inner))


class _RuntimeStub:
    """Enough of MTPLXRuntime for the unbound dispatch under test."""

    def __init__(self, model) -> None:
        self.model = model
        self.counted: list[str] = []

    def _count(self, name: str) -> None:
        self.counted.append(name)

    def forward_ar(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("forward_ar must not be reached by this test")


def test_blocker_names_the_missing_layer_attribute():
    assert generic_hybrid_capture_blocker(_model(with_qwen35_norms=False)) == (
        "input_layernorm"
    )


def test_blocker_is_silent_for_the_families_the_lane_was_written_for():
    assert generic_hybrid_capture_blocker(_model(with_qwen35_norms=True)) is None


def test_capture_raises_the_lane_to_install_not_an_attribute_error():
    with pytest.raises(RuntimeError) as caught:
        MTPLXRuntime.forward_ar_capture(_RuntimeStub(_model(with_qwen35_norms=False)), None)

    message = str(caught.value)
    assert QWEN4_VERIFY_ENV in message, message
    assert "input_layernorm" in message, message


def _write_pack(tmp_path, model_type: str) -> str:
    import json

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return str(tmp_path)


def test_ladder_installs_the_family_lane_the_server_installs(tmp_path, monkeypatch):
    monkeypatch.delenv(QWEN4_VERIFY_ENV, raising=False)
    model = _write_pack(tmp_path, "qwen4_exp")

    assert _apply_family_verify_lane_override(model) == "1"
    import os

    assert os.environ[QWEN4_VERIFY_ENV] == "1"


def test_ladder_leaves_operator_env_alone(tmp_path, monkeypatch):
    monkeypatch.setenv(QWEN4_VERIFY_ENV, "0")

    assert _apply_family_verify_lane_override(_write_pack(tmp_path, "qwen4_exp")) == "0"
    import os

    assert os.environ[QWEN4_VERIFY_ENV] == "0"


def test_ladder_does_not_touch_other_families(tmp_path, monkeypatch):
    monkeypatch.delenv(QWEN4_VERIFY_ENV, raising=False)

    assert _apply_family_verify_lane_override(_write_pack(tmp_path, "qwen3_5_moe")) is None
    import os

    assert QWEN4_VERIFY_ENV not in os.environ


def test_ladder_survives_a_pack_without_config(tmp_path, monkeypatch):
    monkeypatch.delenv(QWEN4_VERIFY_ENV, raising=False)

    assert _apply_family_verify_lane_override(str(tmp_path)) is None


def test_ladder_survives_an_unparseable_config(tmp_path, monkeypatch):
    """A hand-mangled config.json must not break the ladder: no lane, no crash."""

    monkeypatch.delenv(QWEN4_VERIFY_ENV, raising=False)
    (tmp_path / "config.json").write_text("{'model_type': 'qwen4_exp'}", encoding="utf-8")

    assert _apply_family_verify_lane_override(str(tmp_path)) is None
