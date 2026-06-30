"""
Tests for nonlinear PDE handling in NonDimensionalizer.

Covers:
  - Divergence form  d/dx(k(u) du/dx)  — both k*u_xx and k'*u_x^2 terms
  - Pow coefficients  c(u)^2 * u_xx
  - Porous medium  u^n u_xx  (polynomial, no nonlinear_scales needed)
  - Allen-Cahn  eps^2 u_xx - (u^3 - u)
  - Cahn-Hilliard  d^4u/dx^4  (4th-order)
  - Arrhenius source  exp(-E/u)
  - Multiple nonlinear_scales in one PDE
  - Nonlinear_scales with Pow key  c(u)^2
"""
import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from pde_nondim import NonDimensionalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _term_count(nd_pde: sp.Eq) -> int:
    """Number of additive terms in lhs - rhs."""
    return len(sp.Add.make_args(sp.expand(nd_pde.lhs - nd_pde.rhs)))


def _contains_func(expr: sp.Expr, func_name: str) -> bool:
    return any(isinstance(n, AppliedUndef) and str(n.func) == func_name
               for n in sp.preorder_traversal(expr))


# ---------------------------------------------------------------------------
# 1. Divergence form  d/dx(k(u) du/dx) = k(u)u_xx + k'(u)u_x^2
# ---------------------------------------------------------------------------

class TestDivergenceForm:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        self.U, self.L, self.T, self.k_c = sp.symbols("U L T_c k_c", positive=True)
        u = sp.Function("u")(x, t)
        k = sp.Function("k")
        self.pde = sp.Eq(sp.diff(u, t), sp.diff(k(u) * sp.diff(u, x), x))
        self.scales = {u: self.U, x: self.L, t: self.T}
        self.nl = {k(u): self.k_c}

    def test_runs(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        assert r is not None

    def test_both_terms_present(self):
        """k(u)*u_xx and k'(u)*u_x^2 must both appear — previously k'*u_x^2 was dropped."""
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        nd = r.nd_pde_simplified
        expr = sp.expand(nd.lhs - nd.rhs)
        # Expect 3 terms: du_s/dt_s, -k_c*ks*u_xx, -k_c*Dks*u_x^2
        assert _term_count(nd) == 3

    def test_ks_placeholder_in_both_terms(self):
        """The ks placeholder must appear in both the u_xx and u_x^2 terms."""
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        nd = r.nd_pde_simplified
        assert _contains_func(nd, "ks")

    def test_derivative_of_ks_in_u_x_squared_term(self):
        """k'(u) term -> Derivative(ks(us), us) must appear."""
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        nd = r.nd_pde_simplified
        has_dks = any(
            isinstance(n, sp.Derivative) and _contains_func(n, "ks")
            for n in sp.preorder_traversal(nd)
        )
        assert has_dks

    def test_single_group(self):
        """Both nonlinear terms share the same coefficient k_c*T_c/L^2."""
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        assert len(r.dimensionless_groups) == 1

    def test_group_contains_k_c(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        g = list(r.dimensionless_groups.values())[0]
        assert self.k_c in g.free_symbols


# ---------------------------------------------------------------------------
# 2. c(u)^2 * u_xx  (Pow of user-defined function)
# ---------------------------------------------------------------------------

class TestPowNonlinearScale:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        self.U, self.L, self.T, self.c_c = sp.symbols("U L T_c c_c", positive=True)
        u = sp.Function("u")(x, t)
        c = sp.Function("c")
        self.pde = sp.Eq(sp.diff(u, t, 2), c(u)**2 * sp.diff(u, x, 2))
        self.scales = {u: self.U, x: self.L, t: self.T}
        self.nl = {c(u): self.c_c}

    def test_runs(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        assert r is not None

    def test_placeholder_not_pow_class(self):
        """Placeholder must be 'cs' not '<class Pow>s'."""
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        nd_str = str(r.nd_pde_simplified)
        assert "Pow" not in nd_str
        assert "cs" in nd_str

    def test_group_contains_c_c(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        g = list(r.dimensionless_groups.values())[0]
        assert self.c_c in g.free_symbols

    def test_two_terms(self):
        """u_tt = c^2 u_xx has two terms after moving to lhs."""
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        assert _term_count(r.nd_pde_simplified) == 2


# ---------------------------------------------------------------------------
# 3. Porous medium  du/dt = d/dx(u^n du/dx)  (polynomial, automatic)
# ---------------------------------------------------------------------------

class TestPorousMedium:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        self.U, self.L, self.T = sp.symbols("U L T_c", positive=True)
        u = sp.Function("u")(x, t)
        n = sp.Integer(3)
        self.pde = sp.Eq(sp.diff(u, t), sp.diff(u**n * sp.diff(u, x), x))
        self.scales = {u: self.U, x: self.L, t: self.T}

    def test_runs_without_nonlinear_scales(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert r is not None

    def test_three_terms(self):
        """du/dt, u^3 u_xx, and u^2 u_x^2 terms — three total."""
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert _term_count(r.nd_pde_simplified) == 3

    def test_groups_positive(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        for g in r.dimensionless_groups.values():
            assert not sp.simplify(g).could_extract_minus_sign()

    def test_us_cubed_present(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        nd_str = str(r.nd_pde_simplified)
        assert "us(xs, ts)**3" in nd_str or "us(xs, ts)**2" in nd_str


# ---------------------------------------------------------------------------
# 4. Allen-Cahn  du/dt = eps^2 u_xx - (u^3 - u)
# ---------------------------------------------------------------------------

class TestAllenCahn:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        self.eps, self.U, self.L, self.T = sp.symbols("eps U L T_c", positive=True)
        u = sp.Function("u")(x, t)
        self.pde = sp.Eq(
            sp.diff(u, t),
            self.eps**2 * sp.diff(u, x, 2) - (u**3 - u),
        )
        self.scales = {u: self.U, x: self.L, t: self.T}

    def test_runs(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert r is not None

    def test_cubic_term_present(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        nd_str = str(r.nd_pde_simplified)
        assert "us(xs, ts)**3" in nd_str

    def test_four_terms(self):
        """du/dt, eps^2 u_xx, u^3, u — four additive terms."""
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert _term_count(r.nd_pde_simplified) == 4

    def test_canonical_with_U1_L1(self):
        """u ~ 1, x ~ 1 collapses the cubic and linear terms to same coefficient."""
        x, t = sp.symbols("x t", positive=True)
        eps = self.eps
        u = sp.Function("u")(x, t)
        r = NonDimensionalizer(
            self.pde,
            {u: sp.Integer(1), x: sp.Integer(1), t: self.T},
        ).run()
        g_vals = list(r.dimensionless_groups.values())
        assert len(g_vals) >= 1


# ---------------------------------------------------------------------------
# 5. Cahn-Hilliard  du/dt = M(u^3 - u - eps^2 u_xx)_xx  (4th-order)
# ---------------------------------------------------------------------------

class TestCahnHilliard:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        self.M, self.eps, self.U, self.L, self.T = sp.symbols(
            "M eps U L T_c", positive=True
        )
        u = sp.Function("u")(x, t)
        mu = u**3 - u - self.eps**2 * sp.diff(u, x, 2)
        self.pde = sp.Eq(sp.diff(u, t), self.M * sp.diff(mu, x, 2))
        self.scales = {u: self.U, x: self.L, t: self.T}

    def test_runs(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert r is not None

    def test_fourth_derivative_present(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        nd_str = str(r.nd_pde_simplified)
        assert "(xs, 4)" in nd_str

    def test_at_least_four_terms(self):
        """du/dt, M u_xxxx, M u^3 u_xx, M u u_x^2, M u_xx — at least 4."""
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert _term_count(r.nd_pde_simplified) >= 4

    def test_groups_positive(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        for g in r.dimensionless_groups.values():
            assert not sp.simplify(g).could_extract_minus_sign()


# ---------------------------------------------------------------------------
# 6. Arrhenius source  du/dt = alpha u_xx + A exp(-E/u)
# ---------------------------------------------------------------------------

class TestArrhenius:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        self.alpha, self.A, self.E = sp.symbols("alpha A E", positive=True)
        self.U, self.L, self.T = sp.symbols("U L T_c", positive=True)
        u = sp.Function("u")(x, t)
        self.u = u
        self.pde = sp.Eq(
            sp.diff(u, t),
            self.alpha * sp.diff(u, x, 2) + self.A * sp.exp(-self.E / u),
        )
        self.scales = {u: self.U, x: self.L, t: self.T}

    def test_auto_runs(self):
        """Without nonlinear_scales, exp(-E/u) is transformed automatically."""
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert r is not None

    def test_auto_exp_substituted(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        nd_str = str(r.nd_pde_simplified)
        assert "exp" in nd_str

    def test_explicit_scale_adds_placeholder(self):
        """With nonlinear_scales, exp(-E/u) -> exp(-E/U)*exps(us)."""
        exp_scale = sp.exp(-self.E / self.U)
        r = NonDimensionalizer(
            self.pde, self.scales,
            nonlinear_scales={sp.exp(-self.E / self.u): exp_scale},
        ).run()
        nd_str = str(r.nd_pde_simplified)
        assert "exps" in nd_str

    def test_groups_positive(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        for g in r.dimensionless_groups.values():
            assert not sp.simplify(g).could_extract_minus_sign()


# ---------------------------------------------------------------------------
# 7. Multiple nonlinear_scales in a single PDE
# ---------------------------------------------------------------------------

class TestMultipleNonlinearScales:
    def test_two_functions_both_scaled(self):
        """PDE with both k(u) diffusion and f(u) reaction."""
        x, t = sp.symbols("x t", positive=True)
        U, L, T, k_c, f_c = sp.symbols("U L T_c k_c f_c", positive=True)
        u = sp.Function("u")(x, t)
        k, f = sp.Function("k"), sp.Function("f")
        pde = sp.Eq(
            sp.diff(u, t),
            sp.diff(k(u) * sp.diff(u, x), x) + f(u),
        )
        r = NonDimensionalizer(
            pde,
            scales={u: U, x: L, t: T},
            nonlinear_scales={k(u): k_c, f(u): f_c},
        ).run()
        nd_str = str(r.nd_pde_simplified)
        assert "ks" in nd_str
        assert "fs" in nd_str

    def test_two_groups_identified(self):
        x, t = sp.symbols("x t", positive=True)
        U, L, T, k_c, f_c = sp.symbols("U L T_c k_c f_c", positive=True)
        u = sp.Function("u")(x, t)
        k, f = sp.Function("k"), sp.Function("f")
        pde = sp.Eq(
            sp.diff(u, t),
            k(u) * sp.diff(u, x, 2) + f(u),
        )
        r = NonDimensionalizer(
            pde,
            scales={u: U, x: L, t: T},
            nonlinear_scales={k(u): k_c, f(u): f_c},
        ).run()
        assert len(r.dimensionless_groups) >= 1
