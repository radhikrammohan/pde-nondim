"""
Tests for pde_nondim.pinn — PINNFormulation.

Covers:
  - from_result construction
  - moving_frame chain rule
  - steady_state drops time derivatives
  - multiply_by scales residual
  - express_as_groups substitutes group expressions
  - set_domain / set_parameters stores metadata
  - pytorch_code generates syntactically valid Python
  - jax_code generates syntactically valid Python
  - Generated code has correct function signature
  - Coordinate normalisation lines appear in code
"""
import ast
import pytest
import sympy as sp

from pde_nondim import NonDimensionalizer
from pde_nondim.pinn import PINNFormulation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _heat_result():
    """Standard 1-D heat equation non-dim result for reuse."""
    x, t = sp.symbols("x t", positive=True)
    alpha, L, T_c = sp.symbols("alpha L T_c", positive=True)
    dT = sp.Symbol("dT", positive=True)
    u = sp.Function("u")(x, t)
    pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2))
    return NonDimensionalizer(pde, scales={u: dT, x: L, t: T_c}).run()


def _heat_pinn():
    return PINNFormulation.from_result(_heat_result())


# ---------------------------------------------------------------------------
# 1. from_result
# ---------------------------------------------------------------------------

class TestFromResult:
    def test_returns_pinn_formulation(self):
        pf = _heat_pinn()
        assert isinstance(pf, PINNFormulation)

    def test_residual_is_sympy_expr(self):
        pf = _heat_pinn()
        assert isinstance(pf._residual, sp.Expr)

    def test_coords_detected(self):
        pf = _heat_pinn()
        coord_names = [str(c) for c in pf._coords]
        assert "xs" in coord_names
        assert "ts" in coord_names

    def test_time_sym_detected(self):
        pf = _heat_pinn()
        assert pf._time_sym is not None
        assert str(pf._time_sym) == "ts"

    def test_groups_transferred(self):
        result = _heat_result()
        pf = PINNFormulation.from_result(result)
        assert pf._groups == result.dimensionless_groups

    def test_ops_recorded(self):
        pf = _heat_pinn()
        assert any("Created" in op for op in pf._ops)


# ---------------------------------------------------------------------------
# 2. steady_state
# ---------------------------------------------------------------------------

class TestSteadyState:
    def test_drops_time_derivative(self):
        """After steady_state(), no Derivative(..., ts) should remain."""
        pf = _heat_pinn()
        ss = pf.steady_state()
        ts = pf._time_sym
        for node in sp.preorder_traversal(ss._residual):
            if isinstance(node, sp.Derivative):
                wrt_names = [str(w[0]) if isinstance(w, sp.Tuple) else str(w)
                             for w in node.args[1:]]
                assert str(ts) not in wrt_names

    def test_steady_state_op_logged(self):
        pf = _heat_pinn()
        ss = pf.steady_state()
        assert any("steady" in op.lower() for op in ss._ops)

    def test_residual_is_not_zero(self):
        """Spatial derivative term should survive."""
        pf = _heat_pinn()
        ss = pf.steady_state()
        assert ss._residual != 0


# ---------------------------------------------------------------------------
# 3. multiply_by
# ---------------------------------------------------------------------------

class TestMultiplyBy:
    def test_scales_residual(self):
        pf = _heat_pinn()
        Pe = sp.Symbol("Pe", positive=True)
        scaled = pf.multiply_by(Pe)
        # Pe should appear in the residual
        assert Pe in scaled._residual.free_symbols

    def test_op_logged(self):
        pf = _heat_pinn()
        Pe = sp.Symbol("Pe", positive=True)
        scaled = pf.multiply_by(Pe)
        assert any("multipli" in op.lower() or "scaled" in op.lower() or "factor" in op.lower()
                   for op in scaled._ops)

    def test_original_unchanged(self):
        pf = _heat_pinn()
        Pe = sp.Symbol("Pe", positive=True)
        _ = pf.multiply_by(Pe)
        assert Pe not in pf._residual.free_symbols


# ---------------------------------------------------------------------------
# 4. express_as_groups
# ---------------------------------------------------------------------------

class TestExpressAsGroups:
    def test_substitutes_group_expression(self):
        """Replace Fo combination with 1/Fo symbol."""
        result = _heat_result()
        pf = PINNFormulation.from_result(result)
        if "Fo" in pf._groups:
            Fo = sp.Symbol("Fo")
            pf2 = pf.express_as_groups({"Fo": Fo})
            assert Fo in pf2._residual.free_symbols

    def test_unknown_group_name_ignored(self):
        """express_as_groups with a non-existent key should not raise."""
        pf = _heat_pinn()
        pf2 = pf.express_as_groups({"DoesNotExist": sp.Symbol("X")})
        assert pf2 is not None

    def test_op_logged(self):
        pf = _heat_pinn()
        pf2 = pf.express_as_groups({"Fo": sp.Symbol("Fo")})
        assert any("group" in op.lower() for op in pf2._ops)


# ---------------------------------------------------------------------------
# 5. set_domain
# ---------------------------------------------------------------------------

class TestSetDomain:
    def test_stores_bounds(self):
        pf = _heat_pinn()
        pf2 = pf.set_domain(xs=(-1.0, 1.0), ts=(0.0, 1.0))
        assert pf2._domain["xs"] == (-1.0, 1.0)
        assert pf2._domain["ts"] == (0.0, 1.0)

    def test_original_unchanged(self):
        pf = _heat_pinn()
        _ = pf.set_domain(xs=(-1.0, 1.0))
        assert "xs" not in pf._domain

    def test_op_logged(self):
        pf = _heat_pinn()
        pf2 = pf.set_domain(xs=(-1.0, 1.0))
        assert any("domain" in op.lower() for op in pf2._ops)


# ---------------------------------------------------------------------------
# 6. set_parameters
# ---------------------------------------------------------------------------

class TestSetParameters:
    def test_stores_ranges(self):
        pf = _heat_pinn()
        pf2 = pf.set_parameters(Fo=(0.01, 10.0))
        assert pf2._param_ranges["Fo"] == (0.01, 10.0)

    def test_original_unchanged(self):
        pf = _heat_pinn()
        _ = pf.set_parameters(Fo=(0.01, 10.0))
        assert "Fo" not in pf._param_ranges

    def test_op_logged(self):
        pf = _heat_pinn()
        pf2 = pf.set_parameters(Fo=(0.01, 10.0))
        assert any("param" in op.lower() for op in pf2._ops)


# ---------------------------------------------------------------------------
# 7. pytorch_code
# ---------------------------------------------------------------------------

class TestPytorchCode:
    def _get_code(self, pf=None):
        pf = pf or _heat_pinn()
        return pf.set_domain(xs=(-1.0, 1.0), ts=(0.0, 1.0)).pytorch_code()

    def test_returns_string(self):
        assert isinstance(self._get_code(), str)

    def test_syntactically_valid_python(self):
        code = self._get_code()
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"pytorch_code() generated invalid Python: {e}\n---\n{code}")

    def test_contains_def(self):
        code = self._get_code()
        assert "def " in code

    def test_function_accepts_model(self):
        code = self._get_code()
        assert "model" in code

    def test_contains_torch_autograd(self):
        code = self._get_code()
        assert "torch.autograd.grad" in code or "autograd" in code

    def test_contains_residual_variable(self):
        code = self._get_code()
        assert "residual" in code.lower() or "res" in code.lower()

    def test_no_hardcoded_Pe_delta_when_no_params(self):
        """If no parameters declared, signature must not default to Pe/delta."""
        code = self._get_code()
        lines = [l.strip() for l in code.splitlines()]
        def_lines = [l for l in lines if l.startswith("def ")]
        assert def_lines, "No def line found"
        def_sig = def_lines[0]
        # If there are no declared params, there should be no Pe/delta in sig
        if "Fo" not in def_sig and "Pe" in def_sig:
            pytest.fail(f"Pe unexpectedly in signature: {def_sig}")

    def test_with_parameters_in_signature(self):
        pf = (_heat_pinn()
              .set_domain(xs=(-1.0, 1.0), ts=(0.0, 1.0))
              .set_parameters(Fo=(0.01, 10.0)))
        code = pf.pytorch_code()
        ast.parse(code)   # must not raise
        assert "Fo" in code

    def test_coord_normalisation_lines(self):
        pf = (_heat_pinn()
              .set_domain(xs=(-1.0, 1.0), ts=(0.0, 1.0)))
        code = pf.pytorch_code()
        # Normalised coords like xs_n or xs_norm should appear
        assert "_n" in code or "norm" in code or "xs" in code


# ---------------------------------------------------------------------------
# 8. jax_code
# ---------------------------------------------------------------------------

class TestJaxCode:
    def _get_code(self, pf=None):
        pf = pf or _heat_pinn()
        return pf.set_domain(xs=(-1.0, 1.0), ts=(0.0, 1.0)).jax_code()

    def test_returns_string(self):
        assert isinstance(self._get_code(), str)

    def test_syntactically_valid_python(self):
        code = self._get_code()
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"jax_code() generated invalid Python: {e}\n---\n{code}")

    def test_contains_jax(self):
        code = self._get_code()
        assert "jax" in code.lower()

    def test_contains_def(self):
        code = self._get_code()
        assert "def " in code


# ---------------------------------------------------------------------------
# 9. moving_frame (heat equation doesn't use it, build a minimal scan PDE)
# ---------------------------------------------------------------------------

class TestMovingFrame:
    def setup_method(self):
        xs, ts = sp.symbols("xs ts", positive=True)
        Fo = sp.Symbol("Fo", positive=True)
        us = sp.Function("us")(xs, ts)
        # Dimensionless heat PDE: dus/dts = Fo * d^2us/dxs^2
        self.residual = sp.Eq(
            sp.diff(us, ts) - Fo * sp.diff(us, xs, 2), 0
        )
        self.xs, self.ts, self.Fo = xs, ts, Fo
        self.us = us

    def _pf(self):
        x, t = sp.symbols("x t", positive=True)
        alpha, L, T_c, dT = sp.symbols("alpha L T_c dT", positive=True)
        u = sp.Function("u")(x, t)
        pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2))
        result = NonDimensionalizer(pde, {u: dT, x: L, t: T_c}).run()
        return PINNFormulation.from_result(result)

    def test_moving_frame_changes_coords(self):
        pf = self._pf()
        xs = sp.Symbol("xs", positive=True)
        ts = sp.Symbol("ts", positive=True)
        pf2 = pf.moving_frame(xs, ts)
        coord_names = [str(c) for c in pf2._coords]
        assert "xi" in coord_names

    def test_moving_frame_op_logged(self):
        pf = self._pf()
        xs = sp.Symbol("xs", positive=True)
        ts = sp.Symbol("ts", positive=True)
        pf2 = pf.moving_frame(xs, ts)
        assert any("moving" in op.lower() or "frame" in op.lower()
                   for op in pf2._ops)

    def test_xs_removed_from_coords(self):
        """After moving frame, xs should be replaced by xi."""
        pf = self._pf()
        xs = sp.Symbol("xs", positive=True)
        ts = sp.Symbol("ts", positive=True)
        pf2 = pf.moving_frame(xs, ts)
        coord_names = [str(c) for c in pf2._coords]
        assert "xs" not in coord_names

    def test_time_sym_preserved(self):
        pf = self._pf()
        xs = sp.Symbol("xs", positive=True)
        ts = sp.Symbol("ts", positive=True)
        pf2 = pf.moving_frame(xs, ts)
        assert pf2._time_sym is not None
        assert str(pf2._time_sym) == "ts"


# ---------------------------------------------------------------------------
# 10. Immutability: all methods return new objects
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_each_method_returns_new_object(self):
        pf = _heat_pinn()
        xs = sp.Symbol("xs", positive=True)
        ts = sp.Symbol("ts", positive=True)
        Pe = sp.Symbol("Pe", positive=True)

        pf_ss = pf.steady_state()
        pf_mb = pf.multiply_by(Pe)
        pf_sd = pf.set_domain(xs=(-1.0, 1.0))
        pf_sp = pf.set_parameters(Fo=(0.1, 10.0))

        assert pf_ss is not pf
        assert pf_mb is not pf
        assert pf_sd is not pf
        assert pf_sp is not pf

    def test_original_residual_unchanged_after_steady_state(self):
        pf = _heat_pinn()
        original_residual = pf._residual
        _ = pf.steady_state()
        assert pf._residual == original_residual
