"""
Tests for pde_nondim.core — NonDimensionalizer and MultipleScalings.
"""
import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from pde_nondim import NonDimensionalizer, MultipleScalings, NondimResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _groups_positive(result: NondimResult) -> bool:
    """Check that all group values have no leading minus sign."""
    for expr in result.dimensionless_groups.values():
        s = sp.simplify(expr)
        if s.could_extract_minus_sign():
            return False
    return True


def _coeff_of(nd_pde_simplified: sp.Eq, group_name: str, result: NondimResult) -> sp.Expr:
    return sp.simplify(result.dimensionless_groups[group_name])


# ---------------------------------------------------------------------------
# 1. Heat equation  ->  Fourier number
# ---------------------------------------------------------------------------

class TestHeatEquation:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        alpha, L, T, dT = sp.symbols("alpha L T DeltaT", positive=True)
        u = sp.Function("u")(x, t)
        self.pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2))
        self.scales = {u: dT, x: L, t: T}
        self.alpha, self.L, self.T = alpha, L, T
        self.u, self.dT = u, dT

    def test_runs(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert isinstance(r, NondimResult)

    def test_fourier_number_present(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert "Fo" in r.dimensionless_groups

    def test_fourier_number_value(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        Fo = sp.simplify(r.dimensionless_groups["Fo"])
        expected = sp.simplify(self.T * self.alpha / self.L**2)
        assert sp.simplify(Fo - expected) == 0

    def test_fourier_number_positive(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert _groups_positive(r)

    def test_canonical_form_with_diffusive_tc(self):
        """When T = L²/alpha, Fo = 1 and normalised PDE has no free parameters."""
        alpha, L, dT = self.alpha, self.L, self.dT
        x, t = sp.symbols("x t", positive=True)
        u = sp.Function("u")(x, t)
        r = NonDimensionalizer(
            self.pde, {u: dT, x: L, t: L**2 / alpha}
        ).run()
        assert r.dimensionless_groups == {}  # no groups left

    def test_substitution_map_contains_u(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        keys_str = [str(k) for k in r.substitution_map]
        assert any("u" in k for k in keys_str)


# ---------------------------------------------------------------------------
# 2. Advection-diffusion  ->  Péclet number
# ---------------------------------------------------------------------------

class TestAdvectionDiffusion:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        V, D, L, T, C0 = sp.symbols("V D L T C0", positive=True)
        u = sp.Function("u")(x, t)
        self.pde = sp.Eq(sp.diff(u, t) + V * sp.diff(u, x), D * sp.diff(u, x, 2))
        self.scales = {u: C0, x: L, t: T}
        self.V, self.D, self.L = V, D, L
        self.u, self.C0 = u, C0

    def test_peclet_present_convective_tc(self):
        V, D, L, C0 = self.V, self.D, self.L, self.C0
        x, t = sp.symbols("x t", positive=True)
        u = sp.Function("u")(x, t)
        r = NonDimensionalizer(self.pde, {u: C0, x: L, t: L / V}).run()
        names = set(r.dimensionless_groups)
        # Péclet group appears in either forward or inverse form
        assert any("Pe" in n for n in names)

    def test_groups_positive(self):
        V, D, L, C0 = self.V, self.D, self.L, self.C0
        x, t = sp.symbols("x t", positive=True)
        u = sp.Function("u")(x, t)
        r = NonDimensionalizer(self.pde, {u: C0, x: L, t: L / V}).run()
        assert _groups_positive(r)


# ---------------------------------------------------------------------------
# 3. Burgers  ->  Reynolds number, nonlinear u·∂u/∂x
# ---------------------------------------------------------------------------

class TestBurgers:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        nu, L, U = sp.symbols("nu L U", positive=True)
        u = sp.Function("u")(x, t)
        self.pde = sp.Eq(
            sp.diff(u, t) + u * sp.diff(u, x),
            nu * sp.diff(u, x, 2),
        )
        self.scales = {u: U, x: L, t: L / U}
        self.nu, self.L, self.U = nu, L, U

    def test_reynolds_present(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        names = set(r.dimensionless_groups)
        assert "1/Re" in names or "Re" in names

    def test_reynolds_value(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        name = "1/Re" if "1/Re" in r.dimensionless_groups else "Re"
        Re_expr = sp.simplify(r.dimensionless_groups[name])
        expected = sp.simplify(self.nu / (self.L * self.U))
        assert sp.simplify(Re_expr - expected) == 0

    def test_groups_positive(self):
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert _groups_positive(r)

    def test_nonlinear_polynomial_handled(self):
        """u·∂u/∂x is a polynomial nonlinearity — must not raise and must produce result."""
        r = NonDimensionalizer(self.pde, self.scales).run()
        assert r.nd_pde_simplified is not None


# ---------------------------------------------------------------------------
# 4. Reference value — gravity cancellation
# ---------------------------------------------------------------------------

class TestReferenceValue:
    def test_gravity_cancels(self):
        t = sp.Symbol("t", positive=True)
        m, k, g, I = sp.symbols("m k g I", positive=True)
        u = sp.Function("u")(t)
        pde = sp.Eq(m * sp.diff(u, t, 2) + k * u, -m * g)

        u_eq = -m * g / k
        tc = sp.sqrt(m / k)

        r = NonDimensionalizer(pde, scales={u: (I, u_eq), t: tc}).run()

        # The normalised PDE should be  d²ūs/dts² + ūs = 0
        # i.e. no free groups remain
        assert r.dimensionless_groups == {}

    def test_substitution_includes_ref(self):
        t = sp.Symbol("t", positive=True)
        m, k, g, I = sp.symbols("m k g I", positive=True)
        u = sp.Function("u")(t)
        pde = sp.Eq(m * sp.diff(u, t, 2) + k * u, -m * g)
        u_eq = -m * g / k
        tc = sp.sqrt(m / k)
        r = NonDimensionalizer(pde, scales={u: (I, u_eq), t: tc}).run()
        # sub_map value for u should contain the ref term
        vals = list(r.substitution_map.values())
        assert any(u_eq in sp.preorder_traversal(v) for v in vals)


# ---------------------------------------------------------------------------
# 5. Nonlinear user-defined function  k(u)
# ---------------------------------------------------------------------------

class TestNonlinearUserDefined:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        L, T, U, k_c = sp.symbols("L T U k_c", positive=True)
        u = sp.Function("u")(x, t)
        k = sp.Function("k")
        self.pde = sp.Eq(sp.diff(u, t), k(u) * sp.diff(u, x, 2))
        self.scales = {u: U, x: L, t: T}
        self.nl = {k(u): k_c}
        self.k_c, self.L, self.T, self.U = k_c, L, T, U
        self.u, self.k, self.k_u = u, k, k(u)

    def test_runs(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        assert isinstance(r, NondimResult)

    def test_nl_substitution_displayed(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        assert r.nonlinear_substitutions  # non-empty

    def test_placeholder_appears_in_nd_pde(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        nd_str = str(r.nd_pde_simplified)
        assert "ks" in nd_str  # dimensionless placeholder

    def test_parameter_free_with_diffusive_tc(self):
        """With t_c = L²/k_c, PDE should have no dimensionless parameters."""
        k_c, L, U = self.k_c, self.L, self.U
        x, t = sp.symbols("x t", positive=True)
        u = sp.Function("u")(x, t)
        k = sp.Function("k")
        r = NonDimensionalizer(
            self.pde,
            scales={u: U, x: L, t: L**2 / k_c},
            nonlinear_scales={k(u): k_c},
        ).run()
        assert r.dimensionless_groups == {}

    def test_groups_positive(self):
        r = NonDimensionalizer(self.pde, self.scales, nonlinear_scales=self.nl).run()
        assert _groups_positive(r)


# ---------------------------------------------------------------------------
# 6. Nonlinear user-defined reaction  f(u) — two scalings
# ---------------------------------------------------------------------------

class TestNonlinearReaction:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        alpha, L, U, f_c = sp.symbols("alpha L U f_c", positive=True)
        u = sp.Function("u")(x, t)
        f = sp.Function("f")
        self.pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2) + f(u))
        self.alpha, self.L, self.U, self.f_c = alpha, L, U, f_c
        self.nl = {f(u): f_c}
        self.u, self.f_u = u, f(u)

    def test_reaction_scaling_group(self):
        alpha, L, U, f_c = self.alpha, self.L, self.U, self.f_c
        r = NonDimensionalizer(
            self.pde,
            scales={self.u: U, sp.Symbol("x", positive=True): L,
                    sp.Symbol("t", positive=True): U / f_c},
            nonlinear_scales=self.nl,
        ).run()
        names = set(r.dimensionless_groups)
        assert any("Pe" in n for n in names)

    def test_no_duplicate_groups(self):
        """Both du/dt and d²u/dx² share the same coefficient — must appear once."""
        alpha, L, U, f_c = self.alpha, self.L, self.U, self.f_c
        r = NonDimensionalizer(
            self.pde,
            scales={self.u: U, sp.Symbol("x", positive=True): L,
                    sp.Symbol("t", positive=True): L**2 / alpha},
            nonlinear_scales=self.nl,
        ).run()
        values = [sp.simplify(v) for v in r.dimensionless_groups.values()]
        # No two values should be equal
        for i, v1 in enumerate(values):
            for v2 in values[i+1:]:
                assert sp.simplify(v1 - v2) != 0

    def test_groups_positive(self):
        alpha, L, U, f_c = self.alpha, self.L, self.U, self.f_c
        r = NonDimensionalizer(
            self.pde,
            scales={self.u: U, sp.Symbol("x", positive=True): L,
                    sp.Symbol("t", positive=True): U / f_c},
            nonlinear_scales=self.nl,
        ).run()
        assert _groups_positive(r)


# ---------------------------------------------------------------------------
# 7. Transcendental nonlinearity  exp(u) — builtin
# ---------------------------------------------------------------------------

class TestTranscendental:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        alpha, L, T, U = sp.symbols("alpha L T U", positive=True)
        u = sp.Function("u")(x, t)
        self.pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2) + sp.exp(u))
        self.scales = {u: U, x: L, t: T}
        self.u, self.U = u, U

    def test_auto_substitution(self):
        """Without nonlinear_scales, exp(U*us) should appear in nd_pde."""
        r = NonDimensionalizer(self.pde, self.scales).run()
        nd_str = str(r.nd_pde)
        assert "exp" in nd_str

    def test_explicit_scale_creates_placeholder(self):
        """With nonlinear_scales={exp(u): exp(U)}, exps(us) placeholder appears."""
        U = self.U
        r = NonDimensionalizer(
            self.pde,
            self.scales,
            nonlinear_scales={sp.exp(self.u): sp.exp(U)},
        ).run()
        nd_str = str(r.nd_pde_simplified)
        assert "exps" in nd_str

    def test_explicit_scale_groups_positive(self):
        U = self.U
        r = NonDimensionalizer(
            self.pde,
            self.scales,
            nonlinear_scales={sp.exp(self.u): sp.exp(U)},
        ).run()
        assert _groups_positive(r)


# ---------------------------------------------------------------------------
# 8. MultipleScalings
# ---------------------------------------------------------------------------

class TestMultipleScalings:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        V, alpha, L, C0 = sp.symbols("V alpha L C0", positive=True)
        u = sp.Function("u")(x, t)
        self.pde = sp.Eq(sp.diff(u, t) + V * sp.diff(u, x), alpha * sp.diff(u, x, 2))
        self.V, self.alpha, self.L, self.C0 = V, alpha, L, C0
        self.u = u

    def test_returns_two_results(self):
        V, alpha, L, C0 = self.V, self.alpha, self.L, self.C0
        u = self.u
        ms = MultipleScalings(
            pde=self.pde,
            scale_options=[
                {u: C0, sp.Symbol("x", positive=True): L,
                 sp.Symbol("t", positive=True): L / V},
                {u: C0, sp.Symbol("x", positive=True): L,
                 sp.Symbol("t", positive=True): L**2 / alpha},
            ],
        )
        results = ms.run_all()
        assert len(results) == 2

    def test_labels_propagated(self):
        V, alpha, L, C0 = self.V, self.alpha, self.L, self.C0
        u = self.u
        ms = MultipleScalings(
            pde=self.pde,
            scale_options=[
                {u: C0, sp.Symbol("x", positive=True): L,
                 sp.Symbol("t", positive=True): L / V},
                {u: C0, sp.Symbol("x", positive=True): L,
                 sp.Symbol("t", positive=True): L**2 / alpha},
            ],
            labels=["Convective", "Diffusive"],
        )
        labels = [lbl for lbl, _ in ms.run_all()]
        assert labels == ["Convective", "Diffusive"]

    def test_both_scalings_produce_peclet(self):
        """Both scalings reveal the same Péclet number value; each run succeeds."""
        V, alpha, L, C0 = self.V, self.alpha, self.L, self.C0
        u = self.u
        ms = MultipleScalings(
            pde=self.pde,
            scale_options=[
                {u: C0, sp.Symbol("x", positive=True): L,
                 sp.Symbol("t", positive=True): L / V},
                {u: C0, sp.Symbol("x", positive=True): L,
                 sp.Symbol("t", positive=True): L**2 / alpha},
            ],
        )
        results = ms.run_all()
        for _label, r in results:
            assert any("Pe" in n for n in r.dimensionless_groups)
            assert _groups_positive(r)


# ---------------------------------------------------------------------------
# 9. NondimResult str representation
# ---------------------------------------------------------------------------

class TestNondimResultStr:
    def test_str_contains_key_sections(self):
        x, t = sp.symbols("x t", positive=True)
        alpha, L, T, dT = sp.symbols("alpha L T DeltaT", positive=True)
        u = sp.Function("u")(x, t)
        pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2))
        r = NonDimensionalizer(pde, {u: dT, x: L, t: T}).run()
        s = str(r)
        assert "NON-DIMENSIONALISATION RESULT" in s
        assert "Substitutions" in s
        assert "Dimensionless groups" in s
        assert "Dominant-balance" in s
