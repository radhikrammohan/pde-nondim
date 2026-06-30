"""
Identification and naming of dimensionless groups.

Based on Langtangen & Pedersen (2016) §3.4 and the classical dimensional
analysis literature.

Groups are matched heuristically by inspecting which physical scale symbols
appear in the numerator vs denominator of each coefficient.  Both a group
(Pe) and its inverse (1/Pe) are detected and labelled accordingly.
"""

from __future__ import annotations

import sympy as sp
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical descriptions (used in dominant-balance output)
# ---------------------------------------------------------------------------

GROUP_DESCRIPTIONS: Dict[str, str] = {
    "Re":   "Reynolds — inertia / viscous",
    "Pe":   "Péclet (mass) — convection / diffusion",
    "Pe_T": "Péclet (heat) — convection / thermal diffusion",
    "Fo":   "Fourier — diffusion time / imposed time",
    "Da":   "Damköhler II — reaction / diffusion",
    "Da_I": "Damköhler I — reaction / convection",
    "St":   "Strouhal — oscillation / convection",
    "Fr":   "Froude — inertia / gravity",
    "Bi":   "Biot — convective / conductive heat transfer",
    "Le":   "Lewis — thermal diffusivity / mass diffusivity",
    "Pr":   "Prandtl — momentum diffusivity / thermal diffusivity",
    "Sc":   "Schmidt — momentum diffusivity / mass diffusivity",
}

# ---------------------------------------------------------------------------
# Pattern database
# Each row: (name, description, numerator_hints, denominator_hints)
# Hints are lowercase substrings matched against symbol names.
# Both a group and its inverse are tried.
# ---------------------------------------------------------------------------

_KNOWN_GROUPS: List[Tuple[str, str, List[str], List[str]]] = [
    ("Re",   "Reynolds",       ["u", "l"],            ["nu", "mu", "visc"]),
    ("Pe",   "Péclet (mass)",  ["u", "l"],            ["d", "diff"]),
    ("Pe_T", "Péclet (heat)",  ["u", "v", "l", "rho", "cp", "c_p"], ["alpha", "a", "kappa", "k"]),
    ("Fo",   "Fourier",        ["alpha", "a", "t"],   ["l"]),
    ("Da",   "Damköhler II",   ["k", "l"],            ["d", "diff"]),
    ("Da_I", "Damköhler I",    ["k", "l"],            ["u"]),
    ("St",   "Strouhal",       ["l"],                 ["u", "t"]),
    ("Fr",   "Froude",         ["u"],                 ["g", "l"]),
    ("Bi",   "Biot",           ["h", "l"],            ["k", "kt", "kappa"]),
    ("Le",   "Lewis",          ["alpha", "a"],        ["d", "diff"]),
    ("Pr",   "Prandtl",        ["nu", "mu"],          ["alpha", "a"]),
    ("Sc",   "Schmidt",        ["nu", "mu"],          ["d", "diff"]),
]


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _sym_name(sym: sp.Basic) -> str:
    return str(sym).lower()


def _score(numer_syms: set, denom_syms: set, n_hints: list, d_hints: list) -> int:
    score = 0
    for h in n_hints:
        if any(h in s for s in numer_syms):
            score += 1
    for h in d_hints:
        if any(h in s for s in denom_syms):
            score += 1
    return score


def _match_group(expr: sp.Expr) -> Optional[Tuple[str, bool]]:
    """Match expr (or 1/expr) to a known group.

    Returns (name, is_inverse) or None.
    is_inverse=True  →  the coefficient is 1/Group, not the Group itself.
    """
    numer, denom = sp.fraction(sp.simplify(expr))
    n_syms = {_sym_name(s) for s in numer.free_symbols}
    d_syms = {_sym_name(s) for s in denom.free_symbols}

    if not n_syms and not d_syms:
        return None

    best: Optional[str] = None
    best_score = 0
    best_inv = False

    for name, _desc, n_hints, d_hints in _KNOWN_GROUPS:
        fwd = _score(n_syms, d_syms, n_hints, d_hints)
        inv = _score(d_syms, n_syms, n_hints, d_hints)

        if fwd >= inv and fwd > best_score:
            best_score, best, best_inv = fwd, name, False
        elif inv > fwd and inv > best_score:
            best_score, best, best_inv = inv, name, True

    return (best, best_inv) if best_score >= 2 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_dimensionless_groups(
    coefficients: List[sp.Expr],
) -> Dict[str, sp.Expr]:
    """
    Given a list of coefficient expressions from a normalised PDE,
    return a dict mapping dimensionless group names to their expressions.

    Both well-known groups (Pe, Re, …) and their inverses (1/Pe, …) are
    recognised.  Unknown groups are labelled Pi_1, Pi_2, …

    Parameters
    ----------
    coefficients : list of sympy.Expr
        One coefficient per additive term of the normalised PDE.

    Returns
    -------
    dict
        e.g. ``{"1/Re": nu/(L*U), "St": L/(U*T)}``
    """
    groups: Dict[str, sp.Expr] = {}
    used: set = set()
    seen_values: list = []   # track (simplified expr) to skip duplicates

    for coeff in coefficients:
        c = sp.simplify(coeff)
        if not c.free_symbols:          # purely numeric → skip
            continue

        # Skip coefficients that are identical (up to sign) to one already seen
        c_abs = -c if c.could_extract_minus_sign() else c
        if any(sp.simplify(c_abs - v) == 0 for v in seen_values):
            continue
        seen_values.append(c_abs)

        # Try matching positive and negative (sign doesn't change the group)
        match = _match_group(c) or _match_group(-c)

        # Always store the positive form of the coefficient so that group
        # values display without spurious leading minus signs.
        c_pos = -c if sp.simplify(c).could_extract_minus_sign() else c

        if match:
            base, inverted = match
            label = f"1/{base}" if inverted else base
            # Unique-ify if name already used
            candidate, k = label, 2
            while candidate in used:
                candidate = f"{label}_{k}"
                k += 1
            groups[candidate] = c_pos
            used.add(candidate)
        else:
            label = f"Pi_{len(groups) + 1}"
            groups[label] = c_pos
            used.add(label)

    return groups
