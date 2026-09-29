"""Fast checks that need no model weights (run in CI)."""

import pytest

from laya_windows.common import clamp_temperature, confidence_from_probs, render_options
from laya_windows.prompt import PromptMixin


def test_noul_defaults_render_false_then_true():
    assert render_options({"t": "noul", "crit": None}) == [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ]


def test_choice_list_becomes_labels():
    q = PromptMixin._to_internal(
        {"type": "choice", "instructions": "Dept?", "criteria": ["billing", "sales"]}
    )
    assert render_options(q) == ["billing", "sales"]


def test_score_levels_are_indexed():
    q = {"t": "score", "crit": ["low", "high"]}
    assert render_options(q) == ["level 0: low", "level 1: high"]


@pytest.mark.parametrize("bad", [
    {"type": "vote", "instructions": "x"},
    {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
    {"type": "score", "instructions": "x", "criteria": []},
])
def test_invalid_questions_are_rejected(bad):
    with pytest.raises(ValueError):
        PromptMixin._to_internal(bad)


def test_temperature_clamp():
    assert clamp_temperature(0.1006) == 0.5
    assert clamp_temperature(9) == 5.0
    assert clamp_temperature("nan") == 1.0


def test_uniform_distribution_has_zero_confidence():
    import numpy as np

    assert confidence_from_probs(np.array([0.25] * 4), 4) == pytest.approx(0.0)
