"""
PINN-ready reformulation of a non-dimensionalized PDE.

Usage
-----
    result = NonDimensionalizer(pde, scales={...}).run()
    pinn = result.to_pinn()

    pinn = (pinn
            .moving_frame(xs, ts)      # fix source at xi=0
            .steady_state()            # drop d/dts (quasi-steady melt pool)
            .multiply_by(Pe)           # balance residual terms
            .set_domain(xi=(-5, 2), eta=(-3, 3), zeta=(-3, 0))
            .set_parameters(Pe=(1, 60), delta=(0.1, 2.0)))

    print(pinn)                        # human-readable summary
    print(pinn.pytorch_code())         # copy into training script
    print(pinn.jax_code())
    print(pinn.reconstruction_code())  # T = T0 + Tc * Ts
"""

from __future__ import annotations

import math
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sympy as sp
from sympy.core.function import AppliedUndef


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_coords(pde_expr: sp.Expr) -> List[sp.Symbol]:
    """Extract all differentiation variables from a PDE expression."""
    coords: set = set()
    for node in sp.preorder_traversal(pde_expr):
        if isinstance(node, sp.Derivative):
            for sym, _ in node.variable_count:
                if isinstance(sym, sp.Symbol):
                    coords.add(sym)
    return sorted(coords, key=str)


def _collect_nd_funcs(pde_expr: sp.Expr) -> List[sp.Expr]:
    """Return the applied undefined functions that appear in PDE derivatives."""
    funcs: set = set()
    for node in sp.preorder_traversal(pde_expr):
        if isinstance(node, sp.Derivative) and isinstance(node.expr, AppliedUndef):
            funcs.add(node.expr)
    return list(funcs)


def _sym_name(sym) -> str:
    return str(sym)


def _sym_match(a, b) -> bool:
    """Match symbols by name only (ignores assumptions like positive=True)."""
    return str(a) == str(b)


def _transform_moving_frame(
    expr: sp.Expr,
    nd_func: sp.Expr,
    new_func: sp.Expr,
    scan_sym: sp.Symbol,
    time_sym: sp.Symbol,
    xi_sym: sp.Symbol,
) -> sp.Expr:
    """
    Apply moving-frame transformation to expr:

      Derivative nodes of nd_func:
        d/d(scan)  ->  d/dxi
        d/d(time)  ->  d/d(time) - d/dxi   (chain rule)

      All other terms (source terms, coefficients):
        scan_sym -> xi_sym + time_sym
        then powsimp to collapse exp(a)*exp(b) -> exp(a+b)
    """
    # Find the actual scan symbol in the expression (handles assumption mismatch)
    def _find_actual(e, target):
        for s in e.free_symbols:
            if _sym_match(s, target):
                return s
        return None

    actual_scan = _find_actual(expr, scan_sym)
    actual_time = _find_actual(expr, time_sym)

    # Split: collect Derivative nodes of our function vs everything else
    terms = sp.Add.make_args(sp.expand(expr))
    result_terms = []

    for term in terms:
        if any(
            isinstance(n, sp.Derivative) and _sym_match(n.expr.func, nd_func.func)
            for n in sp.preorder_traversal(term)
        ):
            # Term contains a derivative of nd_func — apply chain rule
            result_terms.append(_chain_rule_term(term, nd_func, new_func,
                                                  scan_sym, time_sym, xi_sym))
        else:
            # Source / coefficient term — do direct subs then simplify exponents.
            # Use actual_time (the ts from the expression, with correct assumptions)
            # so that exp(-2*ts^2)*exp(-2*(xi+ts)^2)*exp(4*ts*(xi+ts)) cancels to exp(-2*xi^2).
            ts_for_subs = actual_time or time_sym
            subbed = term
            if actual_scan is not None:
                subbed = subbed.subs(actual_scan, xi_sym + ts_for_subs)
            # Combine exp(a)*exp(b) -> exp(a+b) and simplify
            subbed = sp.powsimp(sp.expand(subbed), force=True)
            result_terms.append(subbed)

    return sp.Add(*result_terms)


def _chain_rule_term(
    term: sp.Expr,
    nd_func: sp.Expr,
    new_func: sp.Expr,
    scan_sym: sp.Symbol,
    time_sym: sp.Symbol,
    xi_sym: sp.Symbol,
) -> sp.Expr:
    """Recursively apply chain rule to a term that contains Derivative nodes."""
    if isinstance(term, sp.Derivative) and _sym_match(term.expr.func, nd_func.func):
        var_count = term.variable_count
        new_vc: list = []
        time_order = 0
        for sym, cnt in var_count:
            if _sym_match(sym, scan_sym):
                new_vc.append((xi_sym, cnt))
            elif _sym_match(sym, time_sym):
                new_vc.append((sym, cnt))
                time_order += cnt
            else:
                new_vc.append((sym, cnt))
        main = sp.Derivative(new_func, *new_vc)
        correction = -time_order * sp.Derivative(new_func, xi_sym)
        return main + correction
    elif isinstance(term, AppliedUndef) and _sym_match(term.func, nd_func.func):
        return new_func
    elif term.is_Atom:
        return term
    else:
        new_args = [_chain_rule_term(a, nd_func, new_func,
                                     scan_sym, time_sym, xi_sym)
                    for a in term.args]
        return term.func(*new_args)


def _drop_time_derivative(expr: sp.Expr, nd_func: sp.Expr, time_sym: sp.Symbol) -> sp.Expr:
    """Set all d/dts terms to zero (quasi-steady state). Matches by name."""
    if (isinstance(expr, sp.Derivative)
            and _sym_match(expr.expr.func, nd_func.func)):
        for sym, _ in expr.variable_count:
            if _sym_match(sym, time_sym):
                return sp.Integer(0)
        return expr
    elif expr.is_Atom:
        return expr
    else:
        new_args = [_drop_time_derivative(a, nd_func, time_sym) for a in expr.args]
        return expr.func(*new_args)


def _deriv_varname(func_name: str, sym: str, order: int) -> str:
    """d2us_dxi2 style variable name for generated code."""
    prefix = "d" if order == 1 else f"d{order}"
    suffix = f"_d{sym}{order}" if order > 1 else f"_d{sym}"
    return f"{prefix}{func_name}{suffix}"


def _expr_to_code(expr: sp.Expr, func_name: str, coord_names: List[str]) -> str:
    """
    Convert a SymPy PDE residual expression to Python arithmetic.

    Replaces Derivative(us, xi, 2) with pre-computed variable names like d2us_dxi2.
    Uses sympy.pycode for everything else.
    """
    # Replace derivatives with placeholder symbols first
    substitutions: Dict[sp.Expr, sp.Symbol] = {}
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative) and isinstance(node.expr, AppliedUndef):
            var_count = node.variable_count
            if len(var_count) == 1:
                sym, order = var_count[0]
                vname = _deriv_varname(func_name, str(sym), order)
            else:
                parts = [f"d{str(s)}" * int(n) for s, n in var_count]
                vname = f"d{func_name}_" + "_".join(parts)
            placeholder = sp.Symbol(vname)
            substitutions[node] = placeholder

    simplified = expr.subs(substitutions)

    # Replace the applied function itself with its name
    for node in sp.preorder_traversal(simplified):
        if isinstance(node, AppliedUndef):
            simplified = simplified.subs(node, sp.Symbol(func_name))
            break

    code = sp.pycode(simplified)
    # Replace math.* trig/exp with torch equivalents (needed for tensor inputs)
    for fn in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs"):
        code = code.replace(f"math.{fn}", f"torch.{fn}")
    # pi is a float constant; keep it from math rather than creating a tensor
    code = code.replace("math.pi", "math.pi")
    return code


def _build_grad_lines(nd_func_name: str, coords: List[str], framework: str) -> List[str]:
    """Generate autograd lines for all needed first and second derivatives."""
    lines: List[str] = []

    if framework == "torch":
        def grad1(f, v):
            return (f"d{f}_d{v} = torch.autograd.grad({f}.sum(), {v}, "
                    f"create_graph=True)[0]")
        def grad2(f, v):
            return (f"d2{f}_d{v}2 = torch.autograd.grad(d{f}_d{v}.sum(), {v}, "
                    f"create_graph=True)[0]")
    else:  # jax
        def grad1(f, v):
            return f"d{f}_d{v} = jax.grad(lambda {v}: {f})({v})"
        def grad2(f, v):
            return f"d2{f}_d{v}2 = jax.grad(lambda {v}: d{f}_d{v})({v})"

    lines.append(f"    # --- First-order derivatives ---")
    for c in coords:
        lines.append(f"    {grad1(nd_func_name, c)}")

    lines.append(f"    # --- Second-order derivatives ---")
    for c in coords:
        lines.append(f"    {grad2(nd_func_name, c)}")

    return lines


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

@dataclass
class PINNFormulation:
    """
    PINN-ready reformulation of a non-dimensionalized PDE.

    Obtain via ``NondimResult.to_pinn()`` and chain transformations:

        pinn = (result.to_pinn()
                .moving_frame(xs, ts)
                .steady_state()
                .multiply_by(Pe)
                .set_domain(xi=(-5, 2), eta=(-3, 3), zeta=(-3, 0))
                .set_parameters(Pe=(1, 60), delta=(0.1, 2.0)))
    """

    # PDE in residual form: R = 0
    _residual: sp.Expr
    # The dimensionless dependent function, e.g. Ts(xi, eta, ts)
    _nd_func: sp.Expr
    # Ordered list of coordinate symbols
    _coords: List[sp.Symbol]
    # Time coordinate (may be absent after steady_state())
    _time_sym: Optional[sp.Symbol]
    # Dimensionless groups {name: sympy expr}
    _groups: Dict[str, sp.Expr]
    # Original scales for reconstruction docs
    _scales: Dict
    # Domain bounds {coord_str: (lo, hi)}
    _domain: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Parameter ranges {param_str: (lo, hi)}  for parametric PINN
    _param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Human-readable log of applied transformations
    _ops: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Classmethod constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_result(cls, result) -> "PINNFormulation":
        """Build a PINNFormulation from a NondimResult."""
        pde_eq = result.nd_pde_simplified
        # Residual: move everything to one side
        residual = sp.expand(pde_eq.lhs - pde_eq.rhs)

        # Find the primary nd function (Ts, cs, …)
        nd_funcs = _collect_nd_funcs(residual)
        nd_func = nd_funcs[0] if nd_funcs else None

        coords = _collect_coords(residual)

        # Identify which coord is "time" — heuristic: last coord alphabetically
        # or the one named ts / t
        time_sym = None
        for c in coords:
            if str(c) in ("ts", "t_s", "t", "tau"):
                time_sym = c
                break
        if time_sym is None and coords:
            # fallback: the coord whose name starts with 't'
            for c in coords:
                if str(c).startswith("t"):
                    time_sym = c
                    break
        if time_sym is None and coords:
            time_sym = coords[-1]

        return cls(
            _residual=residual,
            _nd_func=nd_func,
            _coords=coords,
            _time_sym=time_sym,
            _groups=result.dimensionless_groups,
            _scales=result.substitution_map,
            _ops=["Created from NondimResult"],
        )

    # ------------------------------------------------------------------
    # Transformation methods  (each returns a new PINNFormulation)
    # ------------------------------------------------------------------

    def _copy(self, **overrides) -> "PINNFormulation":
        data = {
            "_residual": self._residual,
            "_nd_func": self._nd_func,
            "_coords": list(self._coords),
            "_time_sym": self._time_sym,
            "_groups": dict(self._groups),
            "_scales": dict(self._scales),
            "_domain": dict(self._domain),
            "_param_ranges": dict(self._param_ranges),
            "_ops": list(self._ops),
        }
        data.update(overrides)
        return PINNFormulation(**data)

    def moving_frame(
        self,
        scan_sym: sp.Symbol,
        time_sym: Optional[sp.Symbol] = None,
        new_name: str = "xi",
    ) -> "PINNFormulation":
        """
        Transform to the laser/source moving frame.

        xi = scan_sym - time_sym   (source is always at xi = 0)

        Chain rule on time derivative:
          d/dt|_xs → d/dt|_xi  −  d/dxi

        Parameters
        ----------
        scan_sym : Symbol
            Dimensionless scanning coordinate (e.g. xs).
        time_sym : Symbol, optional
            Dimensionless time coordinate (e.g. ts).  Defaults to self._time_sym.
        new_name : str
            Name for the new co-moving coordinate.  Default 'xi'.
        """
        time_sym = time_sym or self._time_sym
        xi = sp.Symbol(new_name)

        if self._nd_func is None:
            raise ValueError("Cannot apply moving frame: no dependent function found.")

        # Build new function with xi replacing scan_sym in args (match by name)
        old_args = list(self._nd_func.args)
        new_args = [xi if _sym_match(a, scan_sym) else a for a in old_args]
        new_func = sp.Function(str(self._nd_func.func))(*new_args)

        new_residual = _transform_moving_frame(
            self._residual, self._nd_func, new_func, scan_sym, time_sym, xi
        )
        new_residual = sp.expand(new_residual)

        new_coords = [xi if _sym_match(c, scan_sym) else c for c in self._coords]

        return self._copy(
            _residual=new_residual,
            _nd_func=new_func,
            _coords=new_coords,
            _time_sym=time_sym,
            _ops=self._ops + [
                f"Moving frame applied: {new_name} = {scan_sym} − {time_sym}  "
                f"[source fixed at {new_name}=0]"
            ],
        )

    def steady_state(self) -> "PINNFormulation":
        """
        Drop d/d(time_sym) terms — quasi-steady assumption.

        Valid when the melt pool reaches a steady shape in the moving frame
        (typically after a few beam-transit times).
        """
        if self._time_sym is None:
            return self._copy(_ops=self._ops + ["steady_state: no time coord found (already steady?)"])

        new_residual = _drop_time_derivative(
            self._residual, self._nd_func, self._time_sym
        )
        new_residual = sp.expand(new_residual)

        new_coords = [c for c in self._coords if not _sym_match(c, self._time_sym)]

        return self._copy(
            _residual=new_residual,
            _coords=new_coords,
            _time_sym=None,
            _ops=self._ops + [
                f"Steady state: ∂/∂{self._time_sym} = 0  "
                f"[quasi-steady melt pool in moving frame]"
            ],
        )

    def multiply_by(self, factor: sp.Expr) -> "PINNFormulation":
        """
        Multiply the PDE residual by *factor* to balance term magnitudes.

        For LPBF with Pe >> 1, multiply by Pe so that:
          1/Pe · (diffusion terms)  →  (diffusion terms)  O(1)
          1    · (time term)        →  Pe · (time term)   O(Pe)  [or dropped in SS]
        """
        factor = sp.sympify(factor)
        new_residual = sp.expand(self._residual * factor)
        return self._copy(
            _residual=new_residual,
            _ops=self._ops + [f"PDE multiplied by {factor}  [balances residual terms]"],
        )

    def set_domain(self, **bounds: Tuple[float, float]) -> "PINNFormulation":
        """
        Set training domain bounds for each coordinate.

        Example::

            pinn.set_domain(xi=(-5, 2), eta=(-3, 3), zeta=(-3, 0))

        Bounds are in the dimensionless coordinate system (not physical).
        Each bound (lo, hi) maps to [-1, 1] inside the network.
        """
        new_domain = {**self._domain, **{str(k): v for k, v in bounds.items()}}
        return self._copy(
            _domain=new_domain,
            _ops=self._ops + [f"Domain set: {bounds}"],
        )

    def express_as_groups(self, group_values: Dict[str, sp.Expr]) -> "PINNFormulation":
        """
        Substitute group expressions in the residual with target symbols.

        This converts a residual that still contains dimensional parameters
        (k, rho, Cp, …) into one expressed purely in terms of the named
        dimensionless groups (Pe, delta, …), making the generated code clean.

        Parameters
        ----------
        group_values : dict
            Maps group name → target expression.  The group name must match a
            key in ``self._groups``.

        Example::

            Pe_sym    = sp.Symbol('Pe')
            delta_sym = sp.Symbol('delta')
            pinn = pinn.express_as_groups({
                '1/Pe_T':   sp.Integer(1) / Pe_sym,
                '1/Pe_T_2': 2*delta_sym / (sp.pi**sp.Rational(3,2) * Pe_sym),
                '1/Pe_T_3': delta_sym**2 / Pe_sym,
            })
        """
        new_res = self._residual
        for name, target in group_values.items():
            if name in self._groups:
                new_res = new_res.subs(self._groups[name], target)
        new_res = sp.expand(new_res)
        return self._copy(
            _residual=new_res,
            _ops=self._ops + [
                f"Groups substituted: {list(group_values.keys())}"
            ],
        )

    def set_parameters(self, **ranges: Tuple[float, float]) -> "PINNFormulation":
        """
        Declare which dimensionless groups are network inputs (parametric PINN).

        Example::

            pinn.set_parameters(Pe=(1, 60), delta=(0.1, 2.0))

        The network will take these as additional inputs, normalised to [0, 1].
        """
        new_ranges = {**self._param_ranges, **{str(k): v for k, v in ranges.items()}}
        return self._copy(
            _param_ranges=new_ranges,
            _ops=self._ops + [f"Parameters declared: {list(ranges.keys())}"],
        )

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    @property
    def _nd_func_name(self) -> str:
        if self._nd_func is None:
            return "Ts"
        return str(self._nd_func.func)

    @property
    def _coord_names(self) -> List[str]:
        return [str(c) for c in self._coords]

    def _domain_normalisation_code(self, indent: int = 4) -> str:
        """Code block that normalises dimensionless coords to [-1, 1]."""
        pad = " " * indent
        lines = [f"{pad}# --- Map coordinates to [-1, 1] ---"]
        for name, (lo, hi) in self._domain.items():
            lines.append(
                f"{pad}{name}_n = 2.0 * ({name} - ({lo})) / ({hi} - ({lo})) - 1.0"
            )
        if not self._domain:
            lines.append(f"{pad}# (no domain bounds set — pass raw coordinates to network)")
        return "\n".join(lines)

    def _param_normalisation_code(self, indent: int = 4) -> str:
        """Code block that normalises parameter inputs to [0, 1]."""
        pad = " " * indent
        lines = [f"{pad}# --- Normalise parameters to [0, 1] ---"]
        for name, (lo, hi) in self._param_ranges.items():
            lines.append(f"{pad}{name}_n = ({name} - {lo}) / ({hi} - {lo})")
        if not self._param_ranges:
            lines.append(f"{pad}# (no extra parameters — use set_parameters() to declare them)")
        return "\n".join(lines)

    def _network_call_code(self, model_var: str, indent: int = 4) -> str:
        """Code for the network forward pass."""
        pad = " " * indent
        coord_ns = [f"{c}_n" if c in self._domain else c for c in self._coord_names]
        param_ns = [f"{p}_n" if p in self._param_ranges else p
                    for p in self._param_ranges.keys()]
        all_inputs = coord_ns + param_ns
        inputs_str = ", ".join(all_inputs)
        func = self._nd_func_name
        return (
            f"{pad}{func} = {model_var}(torch.stack([{inputs_str}], dim=-1))"
        )

    def _residual_expr_code(self, indent: int = 4) -> str:
        """Expand the SymPy residual into Python arithmetic."""
        pad = " " * indent
        func = self._nd_func_name
        expr = sp.expand(self._residual)
        code = _expr_to_code(expr, func, self._coord_names)
        # wrap long lines
        lines = [f"{pad}R = ("]
        # split at top-level + / - for readability
        terms = str(code).replace("+ -", "- ").split(" + ")
        for i, t in enumerate(terms):
            comma = "" if i == len(terms) - 1 else " +"
            lines.append(f"{pad}    {t.strip()}{comma}")
        lines.append(f"{pad})")
        return "\n".join(lines)

    def pytorch_code(
        self,
        model_var: str = "model",
        func_name: str = "pde_residual",
    ) -> str:
        """
        Generate a PyTorch PDE residual function.

        Parameters
        ----------
        model_var : str
            Variable name of the neural network model.
        func_name : str
            Name for the generated function.

        Returns
        -------
        str
            Python source code (copy into training script).
        """
        coords = self._coord_names
        params = list(self._param_ranges.keys())
        func = self._nd_func_name

        sig_args = ", ".join(coords + params + [model_var])
        grad_lines = _build_grad_lines(func, coords, "torch")

        lines = [
            f"import math",
            f"import torch",
            f"",
            f"",
            f"def {func_name}({sig_args}):",
            f'    """',
            f"    PDE residual — returns zero when the PDE is exactly satisfied.",
            f"",
            f"    Inputs",
            f"    ------",
        ]
        for c in coords:
            lo_hi = self._domain.get(c, "dimensionless")
            lines.append(f"    {c:12s}: torch.Tensor  — dimensionless coord  {lo_hi}")
        for p in params:
            lo_hi = self._param_ranges.get(p, "dimensionless group")
            lines.append(f"    {p:12s}: float or Tensor  — {lo_hi}")
        lines += [
            f"    {model_var:12s}: nn.Module  — maps inputs to {func}",
            f'    """',
            f"    # Enable gradient computation for coordinates",
        ]
        for c in coords:
            lines.append(f"    {c} = {c}.requires_grad_(True)")
        lines.append("")
        lines.append(self._domain_normalisation_code())
        lines.append("")
        lines.append(self._param_normalisation_code())
        lines.append("")
        lines.append(self._network_call_code(model_var))
        lines.append("")
        lines += grad_lines
        lines.append("")
        lines.append(self._residual_expr_code())
        lines.append("    return R")
        return "\n".join(lines)

    def jax_code(
        self,
        model_var: str = "model",
        func_name: str = "pde_residual",
    ) -> str:
        """Generate a JAX/Flax PDE residual function."""
        coords = self._coord_names
        params = list(self._param_ranges.keys())
        func = self._nd_func_name

        sig_args = ", ".join(coords + params + [model_var, "params"])
        lines = [
            f"import math",
            f"import jax",
            f"import jax.numpy as jnp",
            f"",
            f"",
            f"def {func_name}({sig_args}):",
            f'    """PDE residual for JAX/Flax PINN."""',
        ]
        lines.append(self._domain_normalisation_code())
        lines.append(self._param_normalisation_code())

        coord_ns = [f"{c}_n" if c in self._domain else c for c in coords]
        param_ns = [f"{p}_n" if p in self._param_ranges else p for p in params]
        all_inputs = coord_ns + param_ns
        inputs_str = ", ".join(all_inputs)

        lines += [
            f"    x_in = jnp.stack([{inputs_str}], axis=-1)",
            f"    {func} = {model_var}.apply(params, x_in)",
            f"",
            f"    # Derivatives via jax.jacfwd",
        ]
        for c in coords:
            lines.append(
                f"    dTs_d{c} = jax.jacfwd(lambda {c}: "
                f"{model_var}.apply(params, jnp.stack([{inputs_str}], axis=-1)))({c})"
            )
        lines.append("")
        lines.append(self._residual_expr_code())
        lines.append("    return R")
        return "\n".join(lines)

    def reconstruction_code(self) -> str:
        """Generate the T = T0 + Tc * Ts reconstruction snippet."""
        scale_lines = []
        for orig, expr in self._scales.items():
            scale_lines.append(f"    # {orig} = {expr}")

        return textwrap.dedent(f"""\
        # ---------------------------------------------------------------
        # Reconstruction: dimensional T from PINN output Ts
        # ---------------------------------------------------------------
        def reconstruct_T(xi, eta, zeta, Pe, delta, P, eta_abs, k, r, T0,
                          model):
            \"\"\"
            Recover dimensional temperature from the PINN.

            Parameters
            ----------
            xi, eta, zeta : arrays   dimensionless coords in laser frame
            Pe, delta     : floats   dimensionless groups
            P             : float    laser power [W]
            eta_abs       : float    absorptivity
            k, r          : float    conductivity [W/mK], beam radius [m]
            T0            : float    ambient temperature [K]
            \"\"\"
            # --- Rosenthal temperature scale (absorbs P linearly) ---
            Tc = P * eta_abs / (k * r)

            # --- Query PINN ---
            Ts = model(xi, eta, zeta, Pe=Pe, delta=delta)  # dimensionless

            # --- Reconstruct ---
            T = T0 + Tc * Ts                               # dimensional [K]
            return T

        # Key insight: changing P only multiplies T by a constant.
        # No new PINN evaluation needed — just scale Tc.
        """)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = ["=" * 65, "PINN FORMULATION SUMMARY", "=" * 65]

        lines.append("\nTransformation log:")
        for op in self._ops:
            lines.append(f"  • {op}")

        lines.append("\nDimensionless groups (PINN parameters):")
        for name, expr in self._groups.items():
            lines.append(f"  {name}  =  {sp.simplify(expr)}")

        lines.append("\nNetwork coordinates:")
        for c in self._coords:
            lo_hi = self._domain.get(str(c))
            suffix = f"  ∈ {lo_hi}" if lo_hi else ""
            lines.append(f"  {c}{suffix}")

        if self._param_ranges:
            lines.append("\nNetwork parameters (additional inputs):")
            for p, (lo, hi) in self._param_ranges.items():
                lines.append(f"  {p}  ∈ [{lo}, {hi}]")

        lines.append(f"\nPDE residual (should be zero everywhere in domain):")
        lines.append(f"  R = {sp.expand(self._residual)}")
        lines.append(f"\n  R = 0")

        if self._domain:
            lines.append("\nDomain bounds (dimensionless):")
            for c, (lo, hi) in self._domain.items():
                lines.append(f"  {c:8s} ∈ [{lo:6.1f}, {hi:5.1f}]")
            lines.append("\nCoordinate normalisation (for network input):")
            for c, (lo, hi) in self._domain.items():
                lines.append(f"  {c}_n = 2*({c} - ({lo})) / ({hi} - ({lo})) - 1  →  [-1, 1]")

        lines.append("\nReconstruction:")
        for orig, expr in self._scales.items():
            lines.append(f"  {orig}  =  {expr}")

        lines.append("=" * 65)
        return "\n".join(lines)
