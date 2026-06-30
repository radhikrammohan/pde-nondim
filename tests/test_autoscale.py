"""
Tests for pde_nondim.autoscale — parse_dim and auto_scales.
"""
import pytest
import sympy as sp

from pde_nondim.autoscale import parse_dim, auto_scales


# ---------------------------------------------------------------------------
# 1. parse_dim
# ---------------------------------------------------------------------------

class TestParseDimSI:
    """SI unit strings — the natural way users think about dimensions."""

    def test_kg_per_m3(self):
        assert parse_dim("kg/m^3") == {"M": 1, "L": -3}

    def test_W_per_m_per_K(self):
        assert parse_dim("W/m/K") == {"M": 1, "L": 1, "T": -3, "theta": -1}

    def test_J_per_kg_per_K(self):
        assert parse_dim("J/kg/K") == {"L": 2, "T": -2, "theta": -1}

    def test_Pa(self):
        assert parse_dim("Pa") == {"M": 1, "L": -1, "T": -2}

    def test_W_per_m2_per_K(self):
        assert parse_dim("W/m^2/K") == {"M": 1, "T": -3, "theta": -1}

    def test_kg_per_m_per_s(self):
        assert parse_dim("kg/m/s") == {"M": 1, "L": -1, "T": -1}

    def test_m2_per_s(self):
        assert parse_dim("m^2/s") == {"L": 2, "T": -1}

    def test_W(self):
        assert parse_dim("W") == {"M": 1, "L": 2, "T": -3}

    def test_J(self):
        assert parse_dim("J") == {"M": 1, "L": 2, "T": -2}

    def test_m_per_s(self):
        assert parse_dim("m/s") == {"L": 1, "T": -1}

    def test_K(self):
        assert parse_dim("K") == {"theta": 1}

    def test_m(self):
        assert parse_dim("m") == {"L": 1}

    def test_fundamental_still_works(self):
        assert parse_dim("M/L^3") == {"M": 1, "L": -3}
        assert parse_dim("M*L/T^3/theta") == {"M": 1, "L": 1, "T": -3, "theta": -1}


class TestParseDim:
    def test_single_dim(self):
        assert parse_dim("L") == {"L": 1}

    def test_division(self):
        d = parse_dim("M/L^3")
        assert d == {"M": 1, "L": -3}

    def test_mul_and_div(self):
        d = parse_dim("M*L/T^3/theta")
        assert d["M"] == 1
        assert d["L"] == 1
        assert d["T"] == -3
        assert d["theta"] == -1

    def test_squared(self):
        d = parse_dim("L^2/T^2/theta")
        assert d == {"L": 2, "T": -2, "theta": -1}

    def test_dimensionless_dash(self):
        assert parse_dim("-") == {}

    def test_dimensionless_one(self):
        assert parse_dim("1") == {}

    def test_empty_string(self):
        assert parse_dim("") == {}

    def test_alias_theta(self):
        d = parse_dim("theta")
        assert d == {"theta": 1}

    def test_brackets_stripped(self):
        d = parse_dim("[L]")
        assert d == {"L": 1}

    def test_time_alias_T(self):
        d = parse_dim("T")
        assert d == {"T": 1}

    def test_complex_ratio(self):
        # L^2/T
        d = parse_dim("L^2/T")
        assert d == {"L": 2, "T": -1}


# ---------------------------------------------------------------------------
# 2. auto_scales — heat equation
# ---------------------------------------------------------------------------

class TestAutoScalesHeat:
    def setup_method(self):
        self.x, self.t = sp.symbols("x t", positive=True)
        self.alpha, self.L, self.T0 = sp.symbols("alpha L T_0", positive=True)
        self.T = sp.Function("T")(self.x, self.t)
        self.pde = sp.Eq(
            sp.diff(self.T, self.t),
            self.alpha * sp.diff(self.T, self.x, 2),
        )
        self.dims = {
            self.T:     "theta",
            self.x:     "L",
            self.t:     "T",
            self.alpha: "L^2/T",
            self.L:     "L",
            self.T0:    "theta",
        }

    def test_returns_list(self):
        candidates = auto_scales(self.pde, self.dims)
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_each_entry_has_required_keys(self):
        candidates = auto_scales(self.pde, self.dims)
        for c in candidates:
            assert "scales" in c
            assert "groups" in c
            assert "score" in c
            assert "rank" in c

    def test_ranks_are_1_based_consecutive(self):
        candidates = auto_scales(self.pde, self.dims)
        for i, c in enumerate(candidates, 1):
            assert c["rank"] == i

    def test_rank1_has_lowest_score(self):
        candidates = auto_scales(self.pde, self.dims)
        if len(candidates) > 1:
            assert candidates[0]["score"] <= candidates[1]["score"]

    def test_scales_cover_all_variables(self):
        candidates = auto_scales(self.pde, self.dims)
        c = candidates[0]
        scale_keys_str = {str(k) for k in c["scales"]}
        assert any("T" in k for k in scale_keys_str)   # dependent var
        assert any("x" in k for k in scale_keys_str)   # independent var

    def test_diffusive_time_candidate_present(self):
        """L^2/alpha should appear as a time scale candidate."""
        candidates = auto_scales(self.pde, self.dims)
        scale_exprs = []
        for c in candidates:
            for v, s in c["scales"].items():
                if str(v) == "t":
                    scale_exprs.append(str(sp.simplify(s)))
        diffusive = str(sp.simplify(self.L**2 / self.alpha))
        assert any(diffusive == s or "alpha" in s for s in scale_exprs)

    def test_balanced_diffusive_scores_low(self):
        """With T_c = L^2/alpha, Fo=1 -> score should be 0 (no groups)."""
        num_vals = {self.alpha: 1e-4, self.L: 1.0}
        candidates = auto_scales(self.pde, self.dims, numerical_values=num_vals)
        # The best candidate should have score close to 0 (perfectly balanced)
        assert candidates[0]["score"] < 1.0

    def test_max_candidates_respected(self):
        candidates = auto_scales(self.pde, self.dims, max_candidates=3)
        assert len(candidates) <= 3

    def test_si_unit_input_finds_diffusive_scale(self):
        """Users can supply natural SI units instead of M/L^3 notation."""
        x, t = sp.symbols("x t", positive=True)
        rho, Cp, k, L, dT = sp.symbols("rho C_p k L DeltaT", positive=True)
        T = sp.Function("T")(x, t)
        pde = sp.Eq(rho * Cp * sp.diff(T, t), k * sp.diff(T, x, 2))

        dims_si = {
            T:   "K",
            x:   "m",
            t:   "s",
            rho: "kg/m^3",
            Cp:  "J/kg/K",
            k:   "W/m/K",
            L:   "m",
            dT:  "K",
        }
        num = {rho: 7990, Cp: 500, k: 15, L: 0.01, dT: 1000}
        candidates = auto_scales(pde, dims_si, numerical_values=num)

        # Rank-1 candidate must be the diffusive time scale (score = 0)
        assert candidates[0]["score"] < 0.01
        t_scale = str(sp.simplify(
            [s for v, s in candidates[0]["scales"].items() if str(v) == "t"][0]
        ))
        assert "k" in t_scale or "rho" in t_scale


# ---------------------------------------------------------------------------
# 3. auto_scales — Burgers equation
# ---------------------------------------------------------------------------

class TestAutoScalesBurgers:
    def setup_method(self):
        x, t = sp.symbols("x t", positive=True)
        nu, L, U = sp.symbols("nu L U", positive=True)
        u = sp.Function("u")(x, t)
        self.pde = sp.Eq(
            sp.diff(u, t) + u * sp.diff(u, x),
            nu * sp.diff(u, x, 2),
        )
        self.dims = {
            u:  "L/T",
            x:  "L",
            t:  "T",
            nu: "L^2/T",
            L:  "L",
            U:  "L/T",
        }

    def test_returns_candidates(self):
        candidates = auto_scales(self.pde, self.dims)
        assert len(candidates) > 0

    def test_reynolds_group_appears(self):
        """U*L/nu is the Reynolds number — should appear in some candidate."""
        nu, L, U = sp.symbols("nu L U", positive=True)
        candidates = auto_scales(self.pde, self.dims)
        any_has_re = False
        for c in candidates:
            g_str = " ".join(str(v) for v in c["groups"].values())
            if "nu" in g_str or "Re" in str(c["groups"]):
                any_has_re = True
                break
        assert any_has_re


# ---------------------------------------------------------------------------
# 4. auto_scales — wave equation
# ---------------------------------------------------------------------------

class TestAutoScalesWave:
    def test_wave_speed_appears_as_scale(self):
        x, t = sp.symbols("x t", positive=True)
        c_w, L = sp.symbols("c L", positive=True)
        u = sp.Function("u")(x, t)
        A = sp.Symbol("A", positive=True)
        pde = sp.Eq(sp.diff(u, t, 2), c_w**2 * sp.diff(u, x, 2))
        dims = {
            u:   "L",
            x:   "L",
            t:   "T",
            c_w: "L/T",
            L:   "L",
            A:   "L",
        }
        candidates = auto_scales(pde, dims)
        assert len(candidates) > 0
        # L/c should be a time scale
        scale_strs = []
        for c in candidates:
            for v, s in c["scales"].items():
                if str(v) == "t":
                    scale_strs.append(str(sp.simplify(s)))
        assert any("c" in s for s in scale_strs)

    def test_no_crash_on_minimal_dims(self):
        """auto_scales should not raise even with minimal dim info."""
        x, t = sp.symbols("x t", positive=True)
        u = sp.Function("u")(x, t)
        c_w, L = sp.symbols("c L", positive=True)
        pde = sp.Eq(sp.diff(u, t, 2), c_w**2 * sp.diff(u, x, 2))
        dims = {u: "L", x: "L", t: "T", c_w: "L/T", L: "L"}
        # Must not raise
        candidates = auto_scales(pde, dims)
        assert isinstance(candidates, list)
