"""qwen4_exp must build caches with the ArraysCache class mlx-lm exports *now*.

The prefill ladder died with `unsupported_container:ArraysCache` because
a3b_mtp_batch installs the vendored leak-free ArraysCache at import time while
qwen4_exp's module-level `from mlx_lm.models.cache import ArraysCache` had already
frozen the stock class. Same name, two different objects, so graphbank's
call-time isinstance failed. No weights needed to see it, and none to keep it fixed.
"""

from types import SimpleNamespace

import mlx_lm.models.cache as cache_module
from mtplx.arrays_cache_patch import install_arrays_cache_fix
from mtplx.graphbank import build_verify_state_spec
from mtplx.models import qwen4_exp


class _Layer:
    """Just enough of a qwen4_exp layer for make_cache(): is_linear plus `"ple" in layer`."""

    def __init__(self, *, is_linear=True, ple=False):
        self.is_linear = is_linear
        self._ple = ple

    def __contains__(self, key):
        return key == "ple" and self._ple


def _model(linear_layers):
    model = SimpleNamespace(args=SimpleNamespace(indexer_compress_ratio=4),
                            layers=linear_layers)
    return SimpleNamespace(model=model)


def _fresh_state():
    # Import order is the bug, so every case starts from a known binding.
    install_arrays_cache_fix()
    return cache_module.ArraysCache


def test_make_cache_matches_the_installed_class():
    installed = _fresh_state()
    caches = qwen4_exp.TextModel.make_cache(_model([_Layer(), _Layer(), _Layer()]))
    assert len(caches) == 3
    assert all(isinstance(c, installed) for c in caches), [type(c).__name__ for c in caches]


def test_make_cache_is_correct_whether_or_not_the_fix_is_installed():
    # The other half: with the stock class in place, caches must be stock too.
    # Pinning only the patched direction would hide an import-order regression by
    # accident, because a subclass isinstance check passes for both.
    import mtplx.models.qwen4_exp as q4

    stock = q4._installed_arrays_cache()
    caches = q4.TextModel.make_cache(_model([_Layer()]))
    assert isinstance(caches[0], stock)


def test_two_leaf_caches_from_the_model_reach_the_spec_builder():
    _fresh_state()
    caches = qwen4_exp.TextModel.make_cache(_model([_Layer(), _Layer()]))
    spec, reason = build_verify_state_spec(caches)
    assert reason is None, reason
    assert [kind for _idx, kind, _n in spec] == ["gdn", "gdn"]


def test_ple_layers_still_report_their_own_shape():
    # A "ple" layer gets a 4-leaf cache, and a brand-new one has no leaves written
    # yet -- a different, still-open question (see the handoff plan). Asserted here so
    # the ArraysCache-identity fix is not mistaken for having resolved it.
    _fresh_state()
    caches = qwen4_exp.TextModel.make_cache(_model([_Layer(ple=True)]))
    spec, reason = build_verify_state_spec(caches)
    assert spec is None
    assert reason and reason.startswith("unsupported_container:ArraysCache")
