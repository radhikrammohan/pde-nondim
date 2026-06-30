"""
String / LaTeX convenience parser for PDEs.

Supports a simple derivative notation:
  du/dt       → diff(u, t)
  d2u/dx2     → diff(u, x, 2)
  d(u)/dt     → diff(u, t)
  ∂u/∂t       → diff(u, t)   (unicode partials also accepted)

The parser converts these shorthands to sympy expressions.

Usage
-----
    from pde_nondim.parser import parse_pde
    eq, syms = parse_pde(
        "du/dt = alpha * d2u/dx2",
        functions=['u'],
        variables=['x', 't'],
        parameters=['alpha'],
    )
"""

from __future__ import annotations

import re
import sympy as sp
from typing import List, Optional, Tuple, Dict


def parse_pde(
    pde_str: str,
    functions: List[str],
    variables: List[str],
    parameters: Optional[List[str]] = None,
) -> Tuple[sp.Eq, Dict[str, sp.Basic]]:
    """
    Parse a PDE string into a sympy Eq.

    Parameters
    ----------
    pde_str : str
        PDE as a string, e.g. ``"du/dt = alpha * d2u/dx2 + f"``
    functions : list[str]
        Names of the dependent variable(s), e.g. ``['u']``.
    variables : list[str]
        Names of the independent variables, e.g. ``['x', 't']``.
    parameters : list[str], optional
        Names of parameters, e.g. ``['alpha', 'nu']``.
        Any unrecognised symbol will also be treated as a parameter.

    Returns
    -------
    eq : sympy.Eq
    symbols : dict
        All symbols/functions created, keyed by name.
    """
    parameters = parameters or []

    # --- Build sympy objects ------------------------------------------------
    sym_vars: Dict[str, sp.Symbol] = {
        v: sp.Symbol(v, positive=True) for v in variables
    }
    sym_params: Dict[str, sp.Symbol] = {
        p: sp.Symbol(p, positive=True) for p in parameters
    }
    sym_funcs: Dict[str, sp.AppliedUndef] = {
        f: sp.Function(f)(*[sym_vars[v] for v in variables])
        for f in functions
    }

    all_syms: Dict[str, sp.Basic] = {**sym_vars, **sym_params, **sym_funcs}

    # --- Normalise unicode partials -----------------------------------------
    pde_str = pde_str.replace("∂", "d").replace("∇", "")

    # --- Split on '=' -------------------------------------------------------
    if "=" in pde_str:
        lhs_str, rhs_str = pde_str.split("=", 1)
    else:
        lhs_str, rhs_str = pde_str, "0"

    lhs_expr = _parse_side(lhs_str.strip(), sym_funcs, sym_vars, sym_params)
    rhs_expr = _parse_side(rhs_str.strip(), sym_funcs, sym_vars, sym_params)

    return sp.Eq(lhs_expr, rhs_expr), all_syms


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

# Pattern: d{n?}{func}/d{var}{n?}
# Examples: du/dt, d2u/dx2, d(u)/dt
_DERIV_RE = re.compile(
    r"\bd(\d*)\(?(\w+)\)?/d(\w+?)(\d*)\b"
)


def _parse_side(
    s: str,
    sym_funcs: Dict[str, sp.AppliedUndef],
    sym_vars: Dict[str, sp.Symbol],
    sym_params: Dict[str, sp.Symbol],
) -> sp.Expr:
    """Convert one side of the PDE string to a sympy expression."""
    s = _replace_derivatives(s, sym_funcs, sym_vars)
    # Use Function classes (not applied instances) so that derivative replacement strings
    # like "Derivative(u(x, t), t)" parse correctly.
    func_classes = {name: sp.Function(name) for name in sym_funcs}
    local_ns = {
        **func_classes,
        **sym_vars,
        **sym_params,
        **{k: v for k, v in sp.__dict__.items() if not k.startswith("_")},
    }
    try:
        return sp.sympify(s, locals=local_ns)
    except Exception as exc:
        raise ValueError(
            f"Could not parse expression '{s}': {exc}\n"
            "Hint: ensure all symbols are listed in functions/variables/parameters."
        ) from exc


def _replace_derivatives(
    s: str,
    sym_funcs: Dict[str, sp.AppliedUndef],
    sym_vars: Dict[str, sp.Symbol],
) -> str:
    """Replace 'd2u/dx2' style shorthands with 'Derivative(u(x,t), (x,2))'."""

    def replace_match(m: re.Match) -> str:
        order_prefix = m.group(1)   # digit before func name (d2u → 2)
        func_name = m.group(2)      # u
        var_name = m.group(3)       # x
        order_suffix = m.group(4)   # digit after var name (dx2 → 2)

        order_str = order_prefix or order_suffix or "1"
        try:
            order = int(order_str)
        except ValueError:
            order = 1

        if func_name not in sym_funcs:
            return m.group(0)   # leave unchanged
        if var_name not in sym_vars:
            return m.group(0)

        func_repr = f"{func_name}({', '.join(str(v) for v in sym_vars)})"
        var_repr = sym_vars[var_name]
        if order == 1:
            return f"Derivative({func_repr}, {var_repr})"
        else:
            return f"Derivative({func_repr}, ({var_repr}, {order}))"

    return _DERIV_RE.sub(replace_match, s)
