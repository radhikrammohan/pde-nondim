"""
Scale suggestion via the "balance of terms" principle.

Langtangen & Pedersen (2016) §2.1.3, §3.1, §3.4:
  "Choose tc to make the coefficient in front of any of the spatial derivative
   terms equal unity" — i.e. demand that the ratio of two term coefficients = 1,
   then solve for the unknown scale.

This module takes a PDE whose coefficients are expressed symbolically in
terms of scales (some known, some unknown symbols), and returns:
  - a set of candidate scale assignments obtained by equating pairs of term
    coefficients (dominant-balance approach)
  - the resulting dimensionless groups under each candidate assignment

Usage
-----
    from pde_nondim.scales import suggest_scales

    tc = sp.Symbol('tc', positive=True)
    candidates = suggest_scales(
        pde=pde,
        known_scales={u: C0, x: L, t: tc},  # tc is the unknown — pass it as its own scale
        unknown_scales=[tc],                  # solve for tc
    )
    for desc, assignment in candidates:
        print(desc, assignment)
"""

from __future__ import annotations

import itertools
import sympy as sp
from sympy.core.function import AppliedUndef
from typing import Dict, List, Optional, Tuple


def suggest_scales(
    pde: sp.Eq,
    known_scales: Dict,
    unknown_scales: List[sp.Symbol],
    nd_suffix: str = "s",
) -> List[Tuple[str, Dict[sp.Symbol, sp.Expr]]]:
    """
    Suggest values for unknown characteristic scales by balancing pairs of
    additive terms in the PDE (dominant-balance / term-balance approach).

    For each pair of terms in lhs - rhs, the function solves for the unknown
    scale that makes the two terms' coefficients equal (ratio = 1).

    Parameters
    ----------
    pde : sympy.Eq
        The *dimensional* PDE.
    known_scales : dict
        Already-decided scales, e.g. ``{x: L, u: U}``.
    unknown_scales : list[sympy.Symbol]
        Scale symbols to solve for, e.g. ``[t_c]``.
    nd_suffix : str
        Suffix used to build dimensionless variable names.

    Returns
    -------
    list of (description, assignment_dict)
        Each entry gives a human-readable balance description and a dict
        mapping each unknown scale symbol to its suggested value.
        Solutions that are negative or trivially zero are filtered out.
    """
    from .core import NonDimensionalizer  # avoid circular import at module level

    # Build a parametric scale dict: known scales + unknowns treated as free syms
    all_scales = dict(known_scales)

    # We need to express the PDE terms' coefficients in terms of scale symbols.
    # Strategy: run NonDimensionalizer with the known scales + unknown_scale symbols
    # as scale values for any missing variables in the PDE.
    nd = NonDimensionalizer(pde=pde, scales=all_scales, nd_suffix=nd_suffix)
    nd._build_specs()

    raw_lhs = nd._transform(pde.lhs)
    raw_rhs = nd._transform(pde.rhs)
    expr = sp.expand(raw_lhs - raw_rhs)
    terms = expr.as_ordered_terms()

    if len(terms) < 2:
        return []

    results = []
    seen_solutions: set = set()

    for (i, t1), (j, t2) in itertools.combinations(enumerate(terms), 2):
        c1 = _extract_coeff(t1)
        c2 = _extract_coeff(t2)

        if c1 == 0 or c2 == 0:
            continue

        ratio = sp.simplify(c1 / c2)

        for unk in unknown_scales:
            if ratio.has(unk):
                try:
                    sols = sp.solve(ratio - 1, unk, positive=True)
                except Exception:
                    sols = []

                if not sols:
                    # Try ratio == -1 (signs may differ)
                    try:
                        sols = sp.solve(sp.Abs(ratio) - 1, unk, positive=True)
                    except Exception:
                        sols = []

                for sol in sols:
                    sol_simplified = sp.simplify(sol)
                    if sol_simplified == 0:
                        continue
                    key = str(sol_simplified)
                    if key in seen_solutions:
                        continue
                    seen_solutions.add(key)

                    term_names = _term_label(terms[i]), _term_label(terms[j])
                    desc = (
                        f"Balance terms {i+1} and {j+1} "
                        f"({term_names[0]}  ~  {term_names[1]}): "
                        f"  {unk} = {sol_simplified}"
                    )
                    results.append((desc, {unk: sol_simplified}))

    return results


def _extract_coeff(term: sp.Expr) -> sp.Expr:
    """Extract the parametric coefficient from a scaled term."""
    factors = sp.Mul.make_args(term)
    param_factors = [
        f for f in factors
        if not f.has(AppliedUndef) and not f.has(sp.Derivative)
    ]
    return sp.Mul(*param_factors) if param_factors else sp.Integer(1)


def _term_label(term: sp.Expr) -> str:
    """Short human-readable label for a PDE term."""
    # Show derivatives symbolically
    deriv = None
    for sub in sp.preorder_traversal(term):
        if isinstance(sub, sp.Derivative):
            deriv = sub
            break
    if deriv is not None:
        return str(deriv)
    for sub in sp.preorder_traversal(term):
        if isinstance(sub, AppliedUndef):
            return str(sub)
    return str(term)[:40]
