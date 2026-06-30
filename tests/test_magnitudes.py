"""
Tests for NondimResult.check_magnitudes().
"""
import pytest
import sympy as sp
from pde_nondim import NonDimensionalizer


def _adv_diff_result():
    x, t = sp.symbols("x t", positive=True)
    V, D, L, C0 = sp.symbols("V D L C0", positive=True)
    u = sp.Function("u")(x, t)
    pde = sp.Eq(sp.diff(u, t) + V * sp.diff(u, x), D * sp.diff(u, x, 2))
    return NonDimensionalizer(pde, scales={u: C0, x: L, t: L / V}).run(), V, D, L, C0


class TestCheckMagnitudes:
    def test_returns_string(self):
        r, V, D, L, C0 = _adv_diff_result()
        out = r.check_magnitudes({V: 1.0, D: 1.0, L: 1.0, C0: 1.0})
        assert isinstance(out, str)

    def test_balanced_regime_no_warning(self):
        r, V, D, L, C0 = _adv_diff_result()
        # Pe = V*L/D = 1  -> O(1)
        out = r.check_magnitudes({V: 1.0, D: 1.0, L: 1.0, C0: 1.0})
        assert "O(1)" in out
        assert "well-balanced" in out

    def test_large_group_flagged(self):
        r, V, D, L, C0 = _adv_diff_result()
        # Pe = 1000 >> 1
        out = r.check_magnitudes({V: 10.0, D: 0.01, L: 1.0, C0: 1.0})
        assert "LARGE" in out

    def test_small_group_flagged(self):
        r, V, D, L, C0 = _adv_diff_result()
        # Pe = 0.001 << 1
        out = r.check_magnitudes({V: 0.001, D: 1.0, L: 1.0, C0: 1.0})
        assert "SMALL" in out
        assert "negligible" in out

    def test_tip_shown_for_imbalanced(self):
        r, V, D, L, C0 = _adv_diff_result()
        out = r.check_magnitudes({V: 10.0, D: 0.01, L: 1.0, C0: 1.0})
        assert "suggest_scales" in out

    def test_missing_value_handled(self):
        r, V, D, L, C0 = _adv_diff_result()
        # Omit D — group cannot be evaluated
        out = r.check_magnitudes({V: 1.0, L: 1.0, C0: 1.0})
        assert "cannot evaluate" in out

    def test_custom_threshold(self):
        r, V, D, L, C0 = _adv_diff_result()
        # Pe = 5 — O(1) at threshold=10 but LARGE at threshold=3
        out_default = r.check_magnitudes({V: 5.0, D: 1.0, L: 1.0, C0: 1.0})
        out_strict = r.check_magnitudes({V: 5.0, D: 1.0, L: 1.0, C0: 1.0}, threshold=3.0)
        assert "O(1)" in out_default
        assert "LARGE" in out_strict

    def test_no_groups_message(self):
        # Heat eq with diffusive tc -> Fo=1, no groups remain
        x, t = sp.symbols("x t", positive=True)
        alpha, L, dT = sp.symbols("alpha L DeltaT", positive=True)
        u = sp.Function("u")(x, t)
        pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2))
        r = NonDimensionalizer(pde, {u: dT, x: L, t: L**2 / alpha}).run()
        out = r.check_magnitudes({alpha: 1e-5, L: 0.1})
        assert "No dimensionless groups" in out
