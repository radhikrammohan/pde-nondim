"""
Tests for pde_nondim.groups — identify_dimensionless_groups.
"""
import pytest
import sympy as sp

from pde_nondim.groups import identify_dimensionless_groups, _match_group


class TestMatchGroup:
    def test_reynolds(self):
        U, L, nu = sp.symbols("U L nu", positive=True)
        expr = U * L / nu
        match = _match_group(expr)
        assert match is not None
        name, is_inv = match
        assert name == "Re"
        assert not is_inv

    def test_inverse_reynolds(self):
        U, L, nu = sp.symbols("U L nu", positive=True)
        expr = nu / (U * L)
        match = _match_group(expr)
        assert match is not None
        name, is_inv = match
        assert name == "Re"
        assert is_inv

    def test_fourier(self):
        alpha, T, L = sp.symbols("alpha T L", positive=True)
        expr = alpha * T / L**2
        match = _match_group(expr)
        assert match is not None
        name, is_inv = match
        assert name == "Fo"

    def test_no_match_for_scalar(self):
        assert _match_group(sp.Integer(2)) is None

    def test_no_match_below_threshold(self):
        # Single-hint expression — score < 2, should not match
        L = sp.Symbol("L", positive=True)
        assert _match_group(L) is None


class TestIdentifyGroups:
    def test_empty_input(self):
        assert identify_dimensionless_groups([]) == {}

    def test_purely_numeric_skipped(self):
        result = identify_dimensionless_groups([sp.Integer(1), sp.Rational(1, 2)])
        assert result == {}

    def test_known_group_named(self):
        alpha, T, L = sp.symbols("alpha T L", positive=True)
        result = identify_dimensionless_groups([alpha * T / L**2])
        assert "Fo" in result

    def test_unknown_group_labelled_pi(self):
        a, b = sp.symbols("a b", positive=True)
        result = identify_dimensionless_groups([a * b])
        assert list(result.keys()) == ["Pi_1"]

    def test_no_duplicates(self):
        """Identical coefficients (up to sign) must only appear once."""
        alpha, L, U, f_c = sp.symbols("alpha L U f_c", positive=True)
        c = U * alpha / (L**2 * f_c)
        result = identify_dimensionless_groups([c, -c])
        assert len(result) == 1

    def test_all_values_positive(self):
        """Stored group values must not have a leading minus sign."""
        U, L, nu = sp.symbols("U L nu", positive=True)
        result = identify_dimensionless_groups([-nu / (U * L)])
        for v in result.values():
            assert not sp.simplify(v).could_extract_minus_sign()

    def test_multiple_distinct_groups(self):
        U, L, nu, alpha = sp.symbols("U L nu alpha", positive=True)
        c1 = nu / (U * L)          # 1/Re
        c2 = alpha * sp.Symbol("T", positive=True) / L**2   # Fo
        result = identify_dimensionless_groups([c1, c2])
        assert len(result) == 2
