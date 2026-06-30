"""
Tests for pde_nondim.scales — suggest_scales.
"""
import pytest
import sympy as sp

from pde_nondim import suggest_scales


class TestSuggestScales:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        V, alpha, L, C0 = sp.symbols("V alpha L C0", positive=True)
        self.u = sp.Function("u")(x, t)
        self.pde = sp.Eq(
            sp.diff(self.u, t) + V * sp.diff(self.u, x),
            alpha * sp.diff(self.u, x, 2),
        )
        self.V, self.alpha, self.L, self.C0 = V, alpha, L, C0
        self.tc = sp.Symbol("tc", positive=True)

    def test_returns_list(self):
        candidates = suggest_scales(
            self.pde,
            known_scales={self.u: self.C0, sp.Symbol("x", positive=True): self.L,
                          sp.Symbol("t", positive=True): self.tc},
            unknown_scales=[self.tc],
        )
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_convective_tc_found(self):
        V, L = self.V, self.L
        candidates = suggest_scales(
            self.pde,
            known_scales={self.u: self.C0, sp.Symbol("x", positive=True): L,
                          sp.Symbol("t", positive=True): self.tc},
            unknown_scales=[self.tc],
        )
        values = [sp.simplify(v[self.tc]) for _, v in candidates]
        expected = sp.simplify(L / V)
        assert any(sp.simplify(v - expected) == 0 for v in values)

    def test_diffusive_tc_found(self):
        alpha, L = self.alpha, self.L
        candidates = suggest_scales(
            self.pde,
            known_scales={self.u: self.C0, sp.Symbol("x", positive=True): L,
                          sp.Symbol("t", positive=True): self.tc},
            unknown_scales=[self.tc],
        )
        values = [sp.simplify(v[self.tc]) for _, v in candidates]
        expected = sp.simplify(L**2 / alpha)
        assert any(sp.simplify(v - expected) == 0 for v in values)

    def test_descriptions_are_strings(self):
        candidates = suggest_scales(
            self.pde,
            known_scales={self.u: self.C0, sp.Symbol("x", positive=True): self.L,
                          sp.Symbol("t", positive=True): self.tc},
            unknown_scales=[self.tc],
        )
        for desc, _ in candidates:
            assert isinstance(desc, str)
            assert len(desc) > 0
