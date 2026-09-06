"""One family rule, two entry points.

`mtplx serve` coerces qwen3-next structure verify strategies to `batched` for
qwen4_exp packs; the in-process ladder hardcoded `capture_commit` and paid for it with
qwen4-shaped capture rows lacking `conv_states`, then `KeyError: 'conv_states'` inside
the generic commit (measured on the Flash-Next pack, 2026-09-06). The rule now lives in
`qwen4_fixed_verify.family_verify_strategy` and both callers read it, so the ladder can
no longer measure a lane the product refuses to run.
"""

import argparse
import json
from pathlib import Path

from mtplx.prefill_bench import _ladder_verify_route
from mtplx.qwen4_fixed_verify import (
    FAMILY_VERIFY_STRATEGY,
    QWEN3NEXT_STRUCTURE_VERIFY_STRATEGIES,
    family_verify_strategy,
    model_type_is_qwen4_exp,
)


def _pack(tmp_path: Path, model_type: str, nested: bool = False) -> str:
    cfg = {"model_type": model_type} if not nested else {
        "model_type": "qwen4_exp",
        "text_config": {"model_type": model_type},
    }
    path = tmp_path / model_type.replace(".", "_")
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


def test_reads_the_family_from_config_json(tmp_path):
    assert model_type_is_qwen4_exp(_pack(tmp_path, "qwen4_exp"))
    assert model_type_is_qwen4_exp(_pack(tmp_path, "qwen4_exp_text", nested=True))
    assert not model_type_is_qwen4_exp(_pack(tmp_path, "qwen3_5"))
    # A bare model id, or a directory without config.json, is not an error: no family.
    assert not model_type_is_qwen4_exp("Youssofal/Qwen3.8-Flash-Next")
    assert not model_type_is_qwen4_exp(str(tmp_path / "nowhere"))


def test_structure_lanes_become_batched_for_the_family(tmp_path):
    pack = _pack(tmp_path, "qwen4_exp")
    for strategy in sorted(QWEN3NEXT_STRUCTURE_VERIFY_STRATEGIES):
        assert family_verify_strategy(pack, strategy) == FAMILY_VERIFY_STRATEGY, strategy
    # spelling variants normalize the same way the server's own check does
    assert family_verify_strategy(pack, "CAPTURE-COMMIT") == FAMILY_VERIFY_STRATEGY


def test_other_strategies_and_other_families_are_left_alone(tmp_path):
    pack = _pack(tmp_path, "qwen4_exp")
    assert family_verify_strategy(pack, "sequential") == "sequential"
    assert family_verify_strategy(_pack(tmp_path, "qwen3_5"), "capture_commit") == "capture_commit"
    assert family_verify_strategy("some/model/id", "capture_commit") == "capture_commit"


def test_ladder_route_matches_the_product_for_each_family(tmp_path):
    # qwen4_exp: batched on the default core, i.e. exactly what `mtplx serve` runs.
    assert _ladder_verify_route(_pack(tmp_path, "qwen4_exp")) == ("batched", "stock")
    # everything else keeps the 27B fast path the ladder was written for.
    strategy, core = _ladder_verify_route(_pack(tmp_path, "qwen3_5"))
    assert (strategy, core) == ("capture_commit", "linear-gdn-from-conv-tape")


def test_server_coercion_still_fires(tmp_path, capsys):
    from mtplx.server.openai import _coerce_family_verify_strategy

    args = argparse.Namespace(model=_pack(tmp_path, "qwen4_exp"), verify_strategy="capture_commit")
    _coerce_family_verify_strategy(args)
    assert args.verify_strategy == FAMILY_VERIFY_STRATEGY
    assert "Coercing" in capsys.readouterr().out

    quiet = argparse.Namespace(model=_pack(tmp_path, "qwen3_5"), verify_strategy="capture_commit")
    _coerce_family_verify_strategy(quiet)
    assert quiet.verify_strategy == "capture_commit"
    assert "Coercing" not in capsys.readouterr().out
