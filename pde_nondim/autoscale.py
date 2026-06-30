"""
Automatic characteristic scale discovery via Buckingham Pi / dimensional analysis.

Given a SymPy PDE and the physical dimensions of every symbol, this module:

  1. Parses dimension strings into exponent vectors
     e.g. 'M*L/T^3/theta'  →  {M:1, L:1, T:-3, theta:-1}

  2. Builds the dimensional matrix and finds, for each variable that needs
     a scale, all parameter combinations that have the same dimension
     (solved as a rational linear system — no brute-force enumeration).

  3. Cross-products the per-variable candidates to produce a ranked list of
     complete scale dictionaries suitable for NonDimensionalizer.

  4. Optionally uses numerical parameter values to rank candidates by how
     balanced the resulting PDE coefficients are (all close to O(1) is best).

Usage
-----
    from pde_nondim import auto_scales
    import sympy as sp

    x, t = sp.symbols('x t', positive=True)
    rho, Cp, k, V, r, T0 = sp.symbols('rho C_p k V r T_0', positive=True)
    T = sp.Function('T')(x, t)

    pde = sp.Eq(rho * Cp * sp.diff(T, t), k * sp.diff(T, x, 2))

    dims = {
        T:   'theta',
        x:   'L',
        t:   'T',
        rho: 'M/L^3',
        Cp:  'L^2/T^2/theta',
        k:   'M*L/T^3/theta',
        V:   'L/T',
        r:   'L',
        T0:  'theta',
    }

    candidates = auto_scales(pde, dims)
    for rank, cand in enumerate(candidates, 1):
        print(f"Rank {rank}: {cand['scales']}")
        print(f"  Groups: {cand['groups']}")
        print(f"  Balance score: {cand['score']:.3f}")
"""

from __future__ import annotations

import itertools
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import sympy as sp
from sympy import Matrix, Rational
from sympy.core.function import AppliedUndef


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental dimension set
# ─────────────────────────────────────────────────────────────────────────────

FUND_DIMS = ["M", "L", "T", "theta", "I", "N"]  # mass, length, time, temp, current, amount


# ─────────────────────────────────────────────────────────────────────────────
# Dimension string parser
# ─────────────────────────────────────────────────────────────────────────────

_DIM_ALIASES = {
    # temperature
    "Θ": "theta", "θ": "theta", "K": "theta",
    # time
    "s": "T", "sec": "T",
    # length
    "m": "L",
    # mass
    "kg": "M",
    # current
    "A": "I",
    # amount
    "mol": "N",
    # dimensionless
    "-": "", "1": "", "dimless": "",
}


def parse_dim(s: str) -> Dict[str, int]:
    """
    Parse a dimension string into a dict of {fundamental_dim: exponent}.

    Accepted syntax (case-sensitive for M, L, T, theta, I, N):
        'M/L^3'              →  {M: 1, L: -3}
        'M*L/T^3/theta'      →  {M: 1, L: 1, T: -3, theta: -1}
        'L^2/T^2/theta'      →  {L: 2, T: -2, theta: -1}
        'L'                  →  {L: 1}
        '1'  or  '-'         →  {}   (dimensionless)
    """
    s = s.strip().lstrip("[").rstrip("]")
    # Apply aliases
    for alias, canon in _DIM_ALIASES.items():
        s = re.sub(rf"\b{re.escape(alias)}\b", canon, s)
    s = s.strip()
    if not s or s in ("", "1", "-"):
        return {}

    result: Dict[str, int] = {}

    # Split on * and / keeping the separator
    tokens = re.split(r"([*/])", s)
    sign = 1
    for tok in tokens:
        tok = tok.strip()
        if tok == "*":
            sign = 1
        elif tok == "/":
            sign = -1
        elif tok:
            # Parse base^exponent
            m = re.fullmatch(r"([A-Za-z_]+)\^?([-\d]*)", tok)
            if not m:
                continue
            base, exp_str = m.group(1), m.group(2)
            exp = int(exp_str) if exp_str else 1
            if base:
                result[base] = result.get(base, 0) + sign * exp

    return result


def _dim_vec(dim_dict: Dict[str, int], fund_dims: List[str]) -> List[int]:
    """Convert a dim dict to a fixed-length vector aligned with fund_dims."""
    return [dim_dict.get(d, 0) for d in fund_dims]


# ─────────────────────────────────────────────────────────────────────────────
# Identify variable roles from PDE
# ─────────────────────────────────────────────────────────────────────────────

def _split_roles(pde: sp.Eq, dims: Dict) -> Tuple[List, List, List]:
    """
    Split dims keys into (dep_vars, indep_vars, params).

    dep_vars  : AppliedUndef  (e.g. T(x,t))
    indep_vars: sp.Symbol that appear as arguments of dep_vars
    params    : everything else
    """
    expr = pde.lhs - pde.rhs
    dep_vars, indep_syms = [], set()
    for node in sp.preorder_traversal(expr):
        if isinstance(node, AppliedUndef):
            dep_vars.append(node)
            indep_syms.update(node.args)

    # Match dims keys to roles by name
    dep_names  = {str(d.func) for d in dep_vars}
    indep_names = {str(s) for s in indep_syms}

    dep, indep, param = [], [], []
    for sym, dim in dims.items():
        name = str(sym.func) if isinstance(sym, AppliedUndef) else str(sym)
        if name in dep_names:
            dep.append(sym)
        elif name in indep_names:
            indep.append(sym)
        else:
            param.append(sym)
    return dep, indep, param


# ─────────────────────────────────────────────────────────────────────────────
# Core: find scale expressions for a given target dimension
# ─────────────────────────────────────────────────────────────────────────────

def _find_scales_for_dim(
    target_dim: Dict[str, int],
    param_syms: List[sp.Symbol],
    param_dims: List[Dict[str, int]],
    fund_dims: List[str],
    max_params: int = 3,
) -> List[sp.Expr]:
    """
    Find parameter combinations p1^a1 * p2^a2 * ... whose combined dimension
    equals target_dim, using rational linear algebra.

    Returns a list of SymPy expressions (scale candidates), preferring
    solutions that use fewer parameters and smaller exponents.
    """
    k = len(fund_dims)
    n = len(param_syms)
    if n == 0:
        return []

    b = Matrix([target_dim.get(d, 0) for d in fund_dims])

    # Try subsets of parameters from small to large
    candidates = []
    seen = set()

    for size in range(1, min(max_params, n) + 1):
        for idxs in itertools.combinations(range(n), size):
            sub_syms = [param_syms[i] for i in idxs]
            sub_dims = [param_dims[i] for i in idxs]
            A = Matrix([[d.get(f, 0) for d in sub_dims] for f in fund_dims])

            # Solve A * x = b  (x = exponent vector, rational)
            try:
                sol = A.solve(b)          # unique solution if square & full rank
                if not isinstance(sol, Matrix):
                    sol = Matrix(list(sol.values()))
            except Exception:
                try:
                    sol_full, free_syms = A.gauss_jordan_solve(b)
                    # Take the particular solution (set free vars to 0)
                    sol = sol_full.subs({s: 0 for s in free_syms})
                except Exception:
                    continue

            # Check solution is valid
            if A * sol != b:
                continue

            # Build scale expression
            expr = sp.Mul(*[
                s**sol[j] for j, s in enumerate(sub_syms)
                if sol[j] != 0
            ])
            expr = sp.nsimplify(expr)
            key = str(sp.expand(expr))
            if key not in seen and expr != sp.S.One:
                seen.add(key)
                candidates.append(expr)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Balance scoring
# ─────────────────────────────────────────────────────────────────────────────

def _term_coefficient(term: sp.Expr) -> sp.Expr:
    """
    Extract the scalar prefactor of an additive PDE term, discarding
    Derivative(...) factors.  E.g.:
      -k/(Cp*V*r*rho) * Derivative(Ts, (xs,2))  →  k/(Cp*V*r*rho)
      Derivative(Ts, ts)                          →  1
    """
    factors = sp.Mul.make_args(term)
    coeff_factors = [
        f for f in factors
        if not isinstance(f, sp.Derivative)
        and not (isinstance(f, sp.Pow) and isinstance(f.base, sp.Derivative))
    ]
    return sp.Abs(sp.Mul(*coeff_factors))


def _balance_score(nd_pde: sp.Eq, numerical_values: Optional[Dict]) -> float:
    """
    Score how balanced a non-dimensionalised PDE is.

    Lower is better.  Score = RMS of |log10(|coeff|)| across additive terms,
    where coeff is the scalar prefactor (Derivative factors stripped out).
    A score of 0 means every term has coefficient exactly 1 — perfectly balanced.

    If no numerical_values are supplied, score by the number of free symbols
    remaining in the coefficients (fewer = more self-contained).
    """
    import math
    expr = sp.expand(nd_pde.lhs - nd_pde.rhs)
    terms = sp.Add.make_args(expr)

    if numerical_values:
        num_subs = {sp.Symbol(str(k)): v for k, v in numerical_values.items()}
        log_sq = []
        for term in terms:
            coeff = _term_coefficient(term)
            try:
                val = float(coeff.subs(num_subs))
                if val > 0:
                    log_sq.append(math.log10(val) ** 2)
            except Exception:
                log_sq.append(9.0)   # heavy penalty for un-evaluatable coefficients
        return (sum(log_sq) / len(log_sq)) ** 0.5 if log_sq else 99.0
    else:
        # Count free symbols in coefficients (fewer = simpler / more portable)
        free = set()
        for term in terms:
            free |= _term_coefficient(term).free_symbols
        return float(len(free))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def auto_scales(
    pde: sp.Eq,
    dims: Dict,
    numerical_values: Optional[Dict] = None,
    max_candidates: int = 8,
    max_params_per_scale: int = 3,
) -> List[Dict]:
    """
    Automatically discover candidate characteristic scales for a PDE.

    Parameters
    ----------
    pde : sp.Eq
        The dimensional PDE in SymPy form.
    dims : dict
        Maps each symbol (dependent var, independent var, or parameter)
        to a dimension string.  Dimension strings use M, L, T, theta, I, N.
        Examples:  'M/L^3',  'L^2/T^2/theta',  'L',  'T',  '-' (dimensionless).
    numerical_values : dict, optional
        Maps parameter symbols to representative numerical values.
        Used to rank candidates by coefficient balance.
        If omitted, candidates are ranked by simplicity.
    max_candidates : int
        Maximum number of candidate scale sets to return.
    max_params_per_scale : int
        Maximum number of parameters combined to form each scale.

    Returns
    -------
    list of dict, each with keys:
        'scales' : dict suitable for NonDimensionalizer(pde, scales=...)
        'groups' : dimensionless groups found after non-dimensionalisation
        'score'  : balance score (lower = better)
        'rank'   : 1-based rank
    """
    from .core import NonDimensionalizer

    # Active fundamental dimensions (those that actually appear)
    all_dim_dicts = {sym: parse_dim(str(d)) for sym, d in dims.items()}
    active_fund = sorted({
        fd for dd in all_dim_dicts.values() for fd in dd
        if fd in FUND_DIMS
    })

    dep_vars, indep_vars, param_syms = _split_roles(pde, dims)

    param_dims = [all_dim_dicts[p] for p in param_syms]

    # ── For each variable that needs a scale, find candidates ───────────────
    # Dependent variables: scale is (amplitude, reference)
    # Reference is typically 0 or a known temperature/concentration baseline.
    # We search for the amplitude scale only.
    dep_scale_candidates: Dict[sp.Basic, List] = {}
    for dv in dep_vars:
        target = all_dim_dicts.get(dv, {})
        cands = _find_scales_for_dim(
            target, param_syms, param_dims, active_fund, max_params_per_scale
        )
        # Also include any parameter with the exact same dimension as a scale
        for p, pd in zip(param_syms, param_dims):
            if pd == target:
                expr = p
                if str(expr) not in [str(c) for c in cands]:
                    cands.insert(0, expr)
        dep_scale_candidates[dv] = cands[:max_candidates] or [sp.Integer(1)]

    indep_scale_candidates: Dict[sp.Basic, List] = {}
    for iv in indep_vars:
        target = all_dim_dicts.get(iv, {})
        cands = _find_scales_for_dim(
            target, param_syms, param_dims, active_fund, max_params_per_scale
        )
        for p, pd in zip(param_syms, param_dims):
            if pd == target:
                expr = p
                if str(expr) not in [str(c) for c in cands]:
                    cands.insert(0, expr)
        indep_scale_candidates[iv] = cands[:max_candidates] or [sp.Integer(1)]

    # ── Cross-product: one scale per variable ────────────────────────────────
    all_vars  = dep_vars + indep_vars
    all_cands = [dep_scale_candidates[v] for v in dep_vars] + \
                [indep_scale_candidates[v] for v in indep_vars]

    # Allow up to 4 candidates per variable; total combinations <= 4^n_vars
    # For 3 vars that's 64 combinations — manageable.
    trimmed = [c[:4] for c in all_cands]

    results = []
    seen_scale_keys = set()

    for combo in itertools.product(*trimmed):
        scales = {}
        for var, scale in zip(all_vars, combo):
            if isinstance(var, AppliedUndef):
                # Dependent variable: (amplitude, 0)
                scales[var] = (scale, sp.Integer(0))
            else:
                scales[var] = scale

        key = str(sorted(str(v) for v in scales.values()))
        if key in seen_scale_keys:
            continue
        seen_scale_keys.add(key)

        # Run non-dimensionalisation
        try:
            result = NonDimensionalizer(pde, scales=scales).run()
            nd_pde = result.nd_pde_simplified
            groups = result.dimensionless_groups
            score  = _balance_score(nd_pde, numerical_values)
            results.append({
                "scales": scales,
                "groups": groups,
                "nd_pde": nd_pde,
                "score":  score,
            })
        except Exception:
            continue

        if len(results) >= max_candidates * 3:
            break

    # ── Rank by score ────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["score"])
    results = results[:max_candidates]
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results
