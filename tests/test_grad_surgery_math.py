"""Unit-test scaffold for CRaFT gradient surgery math.

Current phase intentionally provides a skipped test skeleton only.
Future prompts will add concrete numerical assertions.
"""

import pytest


@pytest.mark.skip(reason="TODO: add numerical checks for dot/projection correctness in CRaFT grad surgery")
def test_grad_surgery_projection_math_todo() -> None:
    """TODO:
    1) Construct deterministic toy gradients g_act / g_ret.
    2) Verify projected gradient when dot(g_act, g_ret) < 0.
    3) Verify identity behavior when dot(g_act, g_ret) >= 0.
    """
    assert True
