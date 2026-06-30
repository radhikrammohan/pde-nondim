"""
Tests for pde_nondim.parser — parse_pde.
"""
import pytest
import sympy as sp

from pde_nondim import parse_pde, NonDimensionalizer


class TestParsePde:
    def test_returns_eq_and_syms(self):
        eq, syms = parse_pde(
            "du/dt = alpha * d2u/dx2",
            functions=["u"], variables=["x", "t"], parameters=["alpha"],
        )
        assert isinstance(eq, sp.Eq)
        assert "u" in syms
        assert "x" in syms
        assert "t" in syms

    def test_heat_equation_parsed(self):
        eq, syms = parse_pde(
            "du/dt = alpha * d2u/dx2",
            functions=["u"], variables=["x", "t"], parameters=["alpha"],
        )
        # LHS should contain Derivative w.r.t. t, RHS w.r.t. x twice
        assert eq.lhs.has(sp.Derivative)
        assert eq.rhs.has(sp.Derivative)

    def test_parsed_eq_nondimensionalises(self):
        eq, syms = parse_pde(
            "du/dt = alpha * d2u/dx2",
            functions=["u"], variables=["x", "t"], parameters=["alpha"],
        )
        L, T, dT = sp.symbols("L T DeltaT", positive=True)
        alpha = syms["alpha"]
        r = NonDimensionalizer(
            eq,
            scales={syms["u"]: dT, syms["x"]: L, syms["t"]: T},
        ).run()
        assert "Fo" in r.dimensionless_groups

    def test_unicode_derivative_notation(self):
        eq, syms = parse_pde(
            "∂u/∂t = alpha * ∂²u/∂x²",
            functions=["u"], variables=["x", "t"], parameters=["alpha"],
        )
        assert isinstance(eq, sp.Eq)

    def test_second_derivative_shorthand(self):
        eq, syms = parse_pde(
            "du/dt = alpha * d2u/dx2",
            functions=["u"], variables=["x", "t"], parameters=["alpha"],
        )
        # d2u/dx2 should produce a second-order Derivative
        derivs = [a for a in sp.preorder_traversal(eq) if isinstance(a, sp.Derivative)]
        orders = [sum(int(o) for _, o in d.variable_count) for d in derivs]
        assert 2 in orders
