"""Shape contract of the verify-state spec builder, measured not assumed.

This is why the in-process prefill ladder cannot reach qwen4_exp: `make_cache()`
builds `ArraysCache(size=4)` for "ple" layers, and a freshly constructed one has
all four leaves `None` until the first update writes them. The spec builder calls
that `partial_ple` and refuses to compile, so a ladder run that reaches the
builder before the first ple update dies on a cache that is merely young, not
corrupt. Pinned here because the distinction decides the eventual fix: populate
or defer, do not relax the guard and hand the compiled core None leaves.
"""

from mtplx.graphbank import (
    VERIFY_SPEC_KIND_GDN,
    build_verify_state_spec,
)
from mtplx.vendored_arrays_cache import FixedArraysCache


def ArraysCache(leaves: int):
    """Build the cache class `build_verify_state_spec` will actually see.

    tests/test_arrays_cache_patch.py and two others call install_arrays_cache_fix(), which
    rebinds mlx_lm.models.cache.ArraysCache to the vendored leak-free subclass. A class
    captured at import time is then a different object from the one the spec builder
    isinstance-checks, and the whole file changes meaning depending on test order -- which
    is exactly the confusion this contract test exists to remove.
    """
    from mlx_lm.models.cache import ArraysCache as live
    return live(leaves)


def kinds(spec):
    return [(idx, kind, leaves) for idx, kind, leaves in (spec or [])]


def test_two_leaf_gdn_cache_is_accepted_empty():
    spec, reason = build_verify_state_spec([ArraysCache(2)])
    assert reason is None
    assert kinds(spec) == [(0, VERIFY_SPEC_KIND_GDN, 2)]


def test_fresh_four_leaf_cache_is_rejected_as_partial_ple():
    # The ladder blocker, in one assertion: nothing was ever written to this cache.
    spec, reason = build_verify_state_spec([ArraysCache(4)])
    assert spec is None
    assert reason == "unsupported_container:ArraysCache[partial_ple]"


def test_partially_written_four_leaf_cache_is_also_rejected():
    cache = ArraysCache(4)
    cache.cache[0] = None
    spec, reason = build_verify_state_spec([cache])
    assert spec is None
    assert reason == "unsupported_container:ArraysCache[partial_ple]"


def test_other_leaf_counts_are_unsupported_with_their_own_reason():
    for leaves in (1, 3):
        spec, reason = build_verify_state_spec([ArraysCache(leaves)])
        assert spec is None
        assert reason == f"unsupported_container:ArraysCache[{leaves}]"


def test_two_leaf_cache_with_a_missing_leaf_still_compiles():
    # Asymmetric with the four-leaf rule above: at two leaves a None slot is
    # accepted. Recorded because it is the shape of the fix's blast radius --
    # relaxing the four-leaf case would match what two leaves already allow.
    cache = ArraysCache(2)
    cache.cache[0] = None
    spec, reason = build_verify_state_spec([cache])
    assert reason is None
    assert kinds(spec) == [(0, VERIFY_SPEC_KIND_GDN, 2)]


def test_vendored_fixed_cache_uses_the_same_rule_as_stock():
    spec, reason = build_verify_state_spec([FixedArraysCache(2)])
    assert reason is None
    assert kinds(spec) == [(0, VERIFY_SPEC_KIND_GDN, 2)]


def test_one_bad_entry_poisons_the_whole_layer_list():
    # A qwen4_exp-shaped list: the ple layer alone makes the mixed spec fail,
    # which is what the ladder sees as a single unsupported_container error.
    mixed = [ArraysCache(2), ArraysCache(4), ArraysCache(2)]
    spec, reason = build_verify_state_spec(mixed)
    assert spec is None
    assert reason == "unsupported_container:ArraysCache[partial_ple]"
    clean = [ArraysCache(2), ArraysCache(2), ArraysCache(2)]
    spec, reason = build_verify_state_spec(clean)
    assert reason is None
    assert len(spec or []) == 3
