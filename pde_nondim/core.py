"""
Core non-dimensionalization engine.

Based on: Langtangen & Pedersen, Scaling of Differential Equations (2016).

Nonlinear term handling (section 3.3.2)
----------------------------------------
Three categories are supported:

  (a) Polynomial products — u*du/dx, u**2, u**3 ...
      Handled automatically. The tree-walker distributes scale factors
      through Mul and Pow nodes:
        u**2  ->  U**2 * us**2
        u * diff(u,x)  ->  (U**2/L) * us * Derivative(us, xs)

  (b) User-defined functions  f(u), k(u)  (unknown analytical form)
      Pass nonlinear_scales={k(u): k_c} to the constructor.
      The code produces:  k(u)  ->  k_c * ks(us)
      where ks is a new O(1) dimensionless function.

  (c) Transcendental functions  exp(u), sin(u), log(u) ...
      The argument is automatically substituted:
        exp(u)  ->  exp(U*us + u_ref)
      If the user also passes  nonlinear_scales={exp(u): exp(U)}
      the term is normalised to exactly O(1):
        exp(u)  ->  exp(U) * exp_s(us)   where exp_s(us) = exp(U*us)/exp(U)
"""

from __future__ import annotations

import sympy as sp
from sympy.core.function import AppliedUndef
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from .groups import identify_dimensionless_groups, GROUP_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScaleSpec:
    nd_symbol: sp.Basic
    scale: sp.Expr
    ref_value: sp.Expr = field(default_factory=lambda: sp.Integer(0))


@dataclass
class NlSpec:
    """Scale spec for a nonlinear function f(u)."""
    func_name: str          # e.g. 'k', 'f', 'exp'
    nd_func_name: str       # e.g. 'ks', 'fs', 'exps'
    scale: sp.Expr          # characteristic size of f(u)
    is_builtin: bool        # True for exp/sin/etc., False for k(u)/f(u)
    builtin_func: object    # the sympy function class if is_builtin


@dataclass
class NondimResult:
    nd_pde: sp.Eq
    nd_pde_simplified: sp.Eq
    dimensionless_groups: Dict[str, sp.Expr]
    substitution_map: Dict[sp.Basic, sp.Expr]
    nonlinear_substitutions: Dict[str, sp.Expr]   # f(u) -> f_c * fs(us)
    reference_coefficient: sp.Expr
    dominant_balance: List[str]

    # ------------------------------------------------------------------
    # Numerical magnitude check
    # ------------------------------------------------------------------

    def check_magnitudes(
        self,
        param_values: Dict,
        threshold: float = 10.0,
    ) -> str:
        """Evaluate each dimensionless group numerically and flag outliers.

        Parameters
        ----------
        param_values : dict
            Maps dimensional parameter symbols to their numerical values.
            e.g. ``{L: 0.01, U: 1.0, nu: 1e-6}``
        threshold : float
            Groups with value > threshold are flagged as LARGE;
            groups with value < 1/threshold are flagged as SMALL.
            Default 10 (one order of magnitude from O(1)).

        Returns
        -------
        str
            Formatted report.
        """
        if not self.dimensionless_groups:
            return "No dimensionless groups to evaluate."

        lines = [
            "-" * 65,
            f"MAGNITUDE CHECK  (threshold = {threshold})",
            f"  A group is O(1) if  1/{threshold:.4g} <= value <= {threshold:.4g}",
            "-" * 65,
        ]

        any_warning = False
        for name, expr in self.dimensionless_groups.items():
            try:
                val = float(sp.simplify(expr).subs(param_values))
            except (TypeError, ValueError):
                lines.append(f"  {name:12s}  [cannot evaluate — missing values]")
                continue

            abs_val = abs(val)
            if abs_val > threshold:
                status = f"LARGE  ({val:.3g})  >> 1  — term DOMINATES"
                any_warning = True
            elif abs_val < 1.0 / threshold and abs_val > 0:
                status = f"SMALL  ({val:.3g})  << 1  — term negligible, consider dropping"
                any_warning = True
            elif abs_val == 0:
                status = f"ZERO   ({val:.3g})  — term vanishes exactly"
                any_warning = True
            else:
                status = f"O(1)   ({val:.3g})  ✓"

            lines.append(f"  {name:12s}  {status}")

        if not any_warning:
            lines.append("\n  All groups are O(1) — scaling is well-balanced.")
        else:
            lines.append(
                "\n  Tip: choose scales so that the dominant group equals 1."
            )
            lines.append(
                "  Use suggest_scales() to find the balancing scale automatically."
            )

        lines.append("-" * 65)
        return "\n".join(lines)

    def to_pinn(self):
        """Return a :class:`~pde_nondim.pinn.PINNFormulation` for PINN training.

        Example::

            pinn = result.to_pinn()
            pinn = (pinn
                    .moving_frame(xs, ts)
                    .steady_state()
                    .multiply_by(Pe)
                    .set_domain(xi=(-5, 2), eta=(-3, 3), zeta=(-3, 0))
                    .set_parameters(Pe=(1, 60), delta=(0.1, 2.0)))
            print(pinn)
            print(pinn.pytorch_code())
        """
        from .pinn import PINNFormulation
        return PINNFormulation.from_result(self)

    def __str__(self) -> str:
        lines = ["=" * 65, "NON-DIMENSIONALISATION RESULT", "=" * 65]

        lines.append("\nSubstitutions  (dimensional = ref + scale x dimensionless):")
        for orig, expr in self.substitution_map.items():
            lines.append(f"  {orig}  =  {expr}")

        if self.nonlinear_substitutions:
            lines.append("\nNonlinear function scalings:")
            for orig, nd in self.nonlinear_substitutions.items():
                lines.append(f"  {orig}  ->  {nd}")

        lines.append("\nNon-dimensional PDE (raw, before normalisation):")
        lines.append(f"  {self.nd_pde}")

        lines.append(
            f"\nNormalised by coefficient:  "
            f"{sp.simplify(self.reference_coefficient)}"
        )
        lines.append("\nNon-dimensional PDE  [O(1) normalised]:")
        lines.append(f"  {self.nd_pde_simplified}")

        if self.dimensionless_groups:
            lines.append("\nDimensionless groups:")
            for name, expr in self.dimensionless_groups.items():
                base = name.lstrip("1/").split("_")[0]
                desc = GROUP_DESCRIPTIONS.get(base, "")
                suffix = f"  -- {desc}" if desc else ""
                lines.append(f"  {name}  =  {sp.simplify(expr)}{suffix}")

        if self.dominant_balance:
            lines.append("\nDominant-balance analysis:")
            for note in self.dominant_balance:
                lines.append(f"  * {note}")

        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NonDimensionalizer:
    """Non-dimensionalise a single PDE.

    Parameters
    ----------
    pde : sympy.Eq or sympy.Expr
        The dimensional PDE.
    scales : dict
        {variable: scale}  or  {variable: (scale, ref_value)}.
        Keys: applied functions u(x,t) or plain Symbols x, t.
    nonlinear_scales : dict, optional
        Characteristic scales for nonlinear functions of the dependent
        variable.  Keys are expressions containing u(x,t); values are
        their characteristic sizes.

        Examples::

            k = sp.Function('k')
            nonlinear_scales = {
                k(u):        k_c,        # k(u) -> k_c * ks(us)
                sp.exp(u):   sp.exp(U),  # exp(u) -> exp(U)*exps(us)  O(1)
                sp.sin(u):   sp.Integer(1),  # sin(u) already O(1)
            }

    nd_suffix : str
        Suffix for dimensionless variable names.  Default 's'.
    reference_term : 'first' | 'last' | int
        Which additive term to normalise by.
    """

    def __init__(
        self,
        pde: Union[sp.Eq, sp.Expr],
        scales: Dict,
        nonlinear_scales: Optional[Dict] = None,
        nd_suffix: str = "s",
        reference_term: Union[str, int] = "first",
    ):
        if isinstance(pde, sp.Expr):
            pde = sp.Eq(pde, 0)
        self.pde = pde
        self.nd_suffix = nd_suffix
        self.reference_term = reference_term

        # Normalise scales to (scale, ref_value) pairs
        self._raw_scales: Dict = {}
        for k, v in scales.items():
            if isinstance(v, tuple) and len(v) == 2:
                self._raw_scales[k] = (sp.sympify(v[0]), sp.sympify(v[1]))
            else:
                self._raw_scales[k] = (sp.sympify(v), sp.Integer(0))

        # Nonlinear scales: map from pattern expression -> NlSpec
        self._nl_raw: Dict = nonlinear_scales or {}
        self._nl_specs: List[NlSpec] = []

        self._var_specs: Dict[sp.Symbol, ScaleSpec] = {}
        self._func_specs: Dict[AppliedUndef, ScaleSpec] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> NondimResult:
        self._build_specs()
        self._build_nl_specs()

        raw_lhs = self._transform(self.pde.lhs)
        raw_rhs = self._transform(self.pde.rhs)
        raw_nd_pde = sp.Eq(sp.expand(raw_lhs), sp.expand(raw_rhs))

        simplified_pde, ref_coeff, groups = self._normalize(raw_nd_pde)

        sub_map: Dict = {}
        for sym, spec in self._var_specs.items():
            sub_map[sym] = (
                spec.scale * spec.nd_symbol if spec.ref_value == 0
                else spec.ref_value + spec.scale * spec.nd_symbol
            )
        for func, spec in self._func_specs.items():
            sub_map[func] = (
                spec.scale * spec.nd_symbol if spec.ref_value == 0
                else spec.ref_value + spec.scale * spec.nd_symbol
            )

        nl_subs = self._build_nl_display()
        dom_balance = _dominant_balance_notes(groups)

        return NondimResult(
            nd_pde=raw_nd_pde,
            nd_pde_simplified=simplified_pde,
            dimensionless_groups=groups,
            substitution_map=sub_map,
            nonlinear_substitutions=nl_subs,
            reference_coefficient=ref_coeff,
            dominant_balance=dom_balance,
        )

    # ------------------------------------------------------------------
    # Build specs
    # ------------------------------------------------------------------

    def _nd_name(self, name: str) -> str:
        return name.rstrip("*'") + self.nd_suffix

    def _build_specs(self):
        for sym, (scale, ref) in self._raw_scales.items():
            if isinstance(sym, sp.Symbol):
                nd_sym = sp.Symbol(self._nd_name(str(sym)), positive=True)
                self._var_specs[sym] = ScaleSpec(nd_sym, scale, ref)

        for sym, (scale, ref) in self._raw_scales.items():
            if isinstance(sym, AppliedUndef):
                func_name = str(sym.func)
                nd_args = tuple(
                    self._var_specs[a].nd_symbol if a in self._var_specs else a
                    for a in sym.args
                )
                nd_func = sp.Function(self._nd_name(func_name))(*nd_args)
                self._func_specs[sym] = ScaleSpec(nd_func, scale, ref)

    def _build_nl_specs(self):
        """Parse nonlinear_scales into NlSpec objects."""
        self._nl_specs = []
        for pattern_expr, nl_scale in self._nl_raw.items():
            scale_expr = sp.sympify(nl_scale)
            # Determine function name and whether it is a builtin
            func = pattern_expr.func
            func_name = str(func)
            nd_name = self._nd_name(func_name)
            # Is this a known sympy function (exp, sin, ...) or user-defined?
            is_builtin = not isinstance(pattern_expr, AppliedUndef)
            self._nl_specs.append(NlSpec(
                func_name=func_name,
                nd_func_name=nd_name,
                scale=scale_expr,
                is_builtin=is_builtin,
                builtin_func=func,
            ))

    def _build_nl_display(self) -> Dict[str, sp.Expr]:
        """Build a display dict showing each nonlinear substitution."""
        result = {}
        for spec in self._nl_specs:
            # Show:  f(u)  ->  f_c * fs(us)
            us_nd = next(iter(self._func_specs.values())).nd_symbol if self._func_specs else sp.Symbol("us")
            nd_func = sp.Function(spec.nd_func_name)(us_nd)
            result[f"{spec.func_name}(u)"] = spec.scale * nd_func
        return result

    # ------------------------------------------------------------------
    # Tree-walking transformation
    # ------------------------------------------------------------------

    def _transform(self, expr: sp.Expr) -> sp.Expr:
        # 1. Nonlinear functions first (before general recursion)
        nl = self._try_nl_transform(expr)
        if nl is not None:
            return nl

        # 2. Derivatives — chain rule
        if isinstance(expr, sp.Derivative):
            return self._transform_derivative(expr)

        # 3. Known dependent functions u(x,t) -> u0 + U*us
        if isinstance(expr, AppliedUndef):
            spec = self._find_func_spec(expr)
            if spec is not None:
                return spec.ref_value + spec.scale * spec.nd_symbol
            # Unknown applied function (e.g. k(u)) — recurse into its args.
            # This handles polynomial/transcendental nonlinearities that were
            # NOT listed in nonlinear_scales.
            if expr.args:
                return expr.func(*[self._transform(a) for a in expr.args])
            return expr

        # 4. Independent variable symbols
        if isinstance(expr, sp.Symbol) and expr in self._var_specs:
            spec = self._var_specs[expr]
            return spec.ref_value + spec.scale * spec.nd_symbol

        # 5. Recurse through Mul, Add, Pow, built-in functions (exp, sin, …)
        if expr.args:
            return expr.func(*[self._transform(a) for a in expr.args])

        return expr

    def _try_nl_transform(self, expr: sp.Expr) -> Optional[sp.Expr]:
        """If expr matches a nonlinear scale spec, return scale * nd_version.

        For user-defined f(u):  f(u) -> f_c * fs(us)   [new placeholder func]
        For builtin exp/sin/…:  exp(g(u)) -> scale * exps(us)
            where exps(us) is a new O(1) placeholder and us is extracted
            from the FULL substituted argument.  Only matched when the
            pattern has a single arg that is (or contains) the dependent var.
        """
        for spec in self._nl_specs:
            if spec.is_builtin:
                # Match by function class (exp, sin, …)
                if expr.func is not spec.builtin_func:
                    continue
                # Determine the O(1) placeholder argument: extract the
                # primary dimensionless dependent variable (us).
                us = self._primary_nd_symbol()
                if us is None:
                    continue
                nd_expr = sp.Function(spec.nd_func_name)(us)
                return spec.scale * nd_expr
            else:
                # User-defined: k(u), f(u) — match by function name
                if isinstance(expr, AppliedUndef) and str(expr.func) == spec.func_name:
                    nd_args = [self._transform_inner_func(a) for a in expr.args]
                    nd_expr = sp.Function(spec.nd_func_name)(*nd_args)
                    return spec.scale * nd_expr
                # Also match Pow(f(u), n) when f(u) is in nonlinear_scales
                if (isinstance(expr, sp.Pow)
                        and isinstance(expr.base, AppliedUndef)
                        and str(expr.base.func) == spec.func_name):
                    nd_args = [self._transform_inner_func(a) for a in expr.base.args]
                    nd_base = sp.Function(spec.nd_func_name)(*nd_args)
                    return spec.scale ** expr.exp * nd_base ** expr.exp
        return None

    def _primary_nd_symbol(self) -> Optional[sp.Basic]:
        """Return the dimensionless symbol for the first dependent variable."""
        if self._func_specs:
            return next(iter(self._func_specs.values())).nd_symbol
        return None

    def _transform_derivative(self, deriv: sp.Derivative) -> sp.Expr:
        """Chain rule: d^n u / dx^n  ->  (U/L^n) * d^n us / dxs^n.

        Also handles differentiation with respect to a dependent variable,
        e.g. Derivative(k(u), u) from expanding d/dx(k(u)*du/dx).
        In that case u -> us with scale 1/U, and k(u) -> ks(us) with scale k_c,
        giving (k_c/U) * Derivative(ks(us), us).
        """
        inner = deriv.args[0]
        wrt_args = deriv.args[1:]

        new_inner = self._transform_inner_func(inner)

        scale_factor = sp.Integer(1)
        new_wrt: List = []

        for wrt in wrt_args:
            if isinstance(wrt, sp.Tuple):
                var, order = wrt[0], int(wrt[1])
            else:
                var, order = wrt, 1

            if var in self._var_specs:
                # Differentiation w.r.t. an independent variable (x, t, …)
                vspec = self._var_specs[var]
                scale_factor /= vspec.scale ** order
                nd_var = vspec.nd_symbol
            elif isinstance(var, AppliedUndef):
                # Differentiation w.r.t. a dependent variable, e.g. dk/du.
                # Replace u -> us and contribute 1/U to the scale.
                fspec = self._find_func_spec(var)
                if fspec is not None:
                    scale_factor /= fspec.scale ** order
                    nd_var = fspec.nd_symbol
                else:
                    nd_var = var
            else:
                nd_var = var

            new_wrt.append(nd_var if order == 1 else (nd_var, order))

        func_scale = self._get_func_scale(inner)
        scale_factor *= func_scale

        return sp.expand(scale_factor * sp.Derivative(new_inner, *new_wrt))

    def _transform_inner_func(self, expr: sp.Expr) -> sp.Expr:
        """Replace u(x,t)->us(xs,ts) inside a Derivative (no U scale factor).
        Also maps nonlinear functions k(u)->ks(us) using nonlinear_scales.
        """
        if isinstance(expr, AppliedUndef):
            # 1. Dependent variable (T, u, c, …) -> dimensionless symbol
            spec = self._find_func_spec(expr)
            if spec is not None:
                return spec.nd_symbol
            # 2. Nonlinear function with a registered scale (k(u), f(u), …)
            for nl in self._nl_specs:
                if not nl.is_builtin and str(expr.func) == nl.func_name:
                    nd_args = [self._transform_inner_func(a) for a in expr.args]
                    return sp.Function(nl.nd_func_name)(*nd_args)
            # 3. Unknown function — recursively transform its arguments
            if expr.args:
                return expr.func(*[self._transform_inner_func(a) for a in expr.args])
            return expr

        # Built-in functions (exp, sin, …) inside a Derivative
        for nl in self._nl_specs:
            if nl.is_builtin and expr.func is nl.builtin_func:
                nd_args = [self._transform_inner_func(a) for a in expr.args]
                return expr.func(*nd_args)

        if expr.args:
            return expr.func(*[self._transform_inner_func(a) for a in expr.args])
        return expr

    def _get_func_scale(self, expr: sp.Expr) -> sp.Expr:
        """Return the characteristic scale of an expression.

        Checks dependent-variable specs first, then nonlinear specs,
        then traverses sub-expressions.  Supports Pow(f(u), n) for
        nonlinear_scales keys like c(u)**2.
        """
        if isinstance(expr, AppliedUndef):
            spec = self._find_func_spec(expr)
            if spec:
                return spec.scale
            for nl in self._nl_specs:
                if not nl.is_builtin and str(expr.func) == nl.func_name:
                    return nl.scale
            return sp.Integer(1)
        # Pow(f(u), n) — e.g. c(u)**2 registered as nonlinear_scales key
        if isinstance(expr, sp.Pow) and isinstance(expr.base, AppliedUndef):
            for nl in self._nl_specs:
                if not nl.is_builtin and str(expr.base.func) == nl.func_name:
                    return nl.scale ** expr.exp
        for sub in sp.preorder_traversal(expr):
            if isinstance(sub, AppliedUndef):
                spec = self._find_func_spec(sub)
                if spec:
                    return spec.scale
                for nl in self._nl_specs:
                    if not nl.is_builtin and str(sub.func) == nl.func_name:
                        return nl.scale
        return sp.Integer(1)

    def _find_func_spec(self, expr: AppliedUndef) -> Optional[ScaleSpec]:
        for key, spec in self._func_specs.items():
            if str(expr.func) == str(key.func):
                return spec
        return None

    # ------------------------------------------------------------------
    # O(1) normalisation
    # ------------------------------------------------------------------

    def _normalize(
        self, nd_pde: sp.Eq
    ) -> Tuple[sp.Eq, sp.Expr, Dict[str, sp.Expr]]:
        expr = sp.expand(nd_pde.lhs - nd_pde.rhs)
        terms = expr.as_ordered_terms()

        if not terms:
            return nd_pde, sp.Integer(1), {}

        if self.reference_term == "first":
            idx = 0
        elif self.reference_term == "last":
            idx = -1
        elif isinstance(self.reference_term, int):
            idx = self.reference_term
        else:
            idx = 0

        # Collect dimensionless coordinate symbols so they are excluded
        # from the parametric coefficient in _extract_coeff.
        # This prevents Gaussian/exponential source shapes from being
        # absorbed into the coefficient.
        nd_syms: set = set()
        for spec in self._var_specs.values():
            nd_syms.add(spec.nd_symbol)
        for spec in self._func_specs.values():
            nd_syms.update(spec.nd_symbol.free_symbols)

        ref_coeff = sp.simplify(_extract_coeff(terms[idx], nd_syms))
        if ref_coeff == 0 or ref_coeff is sp.nan:
            ref_coeff = sp.Integer(1)

        norm_expr = sp.simplify(expr / ref_coeff)
        norm_pde = sp.Eq(sp.expand(norm_expr), 0)

        expanded = sp.expand(norm_expr)
        coeff_list = [
            sp.simplify(_extract_coeff(t, nd_syms)) for t in expanded.as_ordered_terms()
        ]
        groups = identify_dimensionless_groups(coeff_list)

        return norm_pde, ref_coeff, groups


# ---------------------------------------------------------------------------
# Multiple-scaling comparison
# ---------------------------------------------------------------------------

class MultipleScalings:
    """Run non-dimensionalisation for several candidate scale sets and compare.

    Parameters
    ----------
    pde : sympy.Eq
    scale_options : list of dict
        Each dict is a ``scales`` argument to ``NonDimensionalizer``.
    nonlinear_scales_options : list of dict, optional
        One nonlinear_scales dict per scaling (or a single dict applied to all).
    labels : list of str, optional
    nd_suffix : str
    reference_term : str | int | list
        Single value applied to all, or one per scaling.
    """

    def __init__(
        self,
        pde: sp.Eq,
        scale_options: List[Dict],
        nonlinear_scales_options: Optional[Union[Dict, List[Dict]]] = None,
        labels: Optional[List[str]] = None,
        nd_suffix: str = "s",
        reference_term: Union[str, int, List] = "first",
    ):
        self.pde = pde
        self.scale_options = scale_options
        self.labels = labels or [f"Scaling {i+1}" for i in range(len(scale_options))]
        self.nd_suffix = nd_suffix

        n = len(scale_options)
        if isinstance(reference_term, list):
            self.reference_terms = reference_term
        else:
            self.reference_terms = [reference_term] * n

        if nonlinear_scales_options is None:
            self.nl_options = [{}] * n
        elif isinstance(nonlinear_scales_options, dict):
            self.nl_options = [nonlinear_scales_options] * n
        else:
            self.nl_options = nonlinear_scales_options

    def run_all(self) -> List[Tuple[str, NondimResult]]:
        results = []
        for label, scales, nl_scales, ref in zip(
            self.labels, self.scale_options, self.nl_options, self.reference_terms
        ):
            nd = NonDimensionalizer(
                pde=self.pde,
                scales=scales,
                nonlinear_scales=nl_scales,
                nd_suffix=self.nd_suffix,
                reference_term=ref,
            )
            results.append((label, nd.run()))
        return results

    def print_all(self):
        for label, result in self.run_all():
            print(f"\n{'#' * 65}")
            print(f"  {label}")
            print(f"{'#' * 65}")
            print(result)


# ---------------------------------------------------------------------------
# Dominant-balance analysis
# ---------------------------------------------------------------------------

def _dominant_balance_notes(groups: Dict[str, sp.Expr]) -> List[str]:
    notes = []
    for name, expr in groups.items():
        base = name.lstrip("1/").split("_")[0]
        inv = name.startswith("1/")
        desc = GROUP_DESCRIPTIONS.get(base, "")
        suffix = f" ({desc})" if desc else ""
        if inv:
            notes.append(
                f"Coefficient 1/{base}{suffix}:"
                f"  -> 0 when {base} >> 1 (term negligible);"
                f"  -> large when {base} << 1 (term dominates)"
            )
        else:
            notes.append(
                f"Coefficient {name}{suffix}:"
                f"  -> large when {name} >> 1 (term dominates);"
                f"  -> 0 when {name} << 1 (term negligible)"
            )
    return notes


# ---------------------------------------------------------------------------
# Shared coefficient extractor
# ---------------------------------------------------------------------------

def _extract_coeff(term: sp.Expr, nd_syms: Optional[set] = None) -> sp.Expr:
    """Extract the purely parametric (scale-symbol) coefficient from a term.

    A factor is 'parametric' if it contains:
      - no AppliedUndef  (dimensionless functions us, ks, ...)
      - no Derivative
      - no dimensionless coordinate symbols (xs, ts, ys, ...)

    The last rule is critical for source terms like
    Q0 * exp(-2*(xs-ts)^2) — the Gaussian is an O(1) shape function
    of the dimensionless coords, not a parametric coefficient.

    Parameters
    ----------
    term : sympy.Expr
    nd_syms : set of sympy.Symbol, optional
        The dimensionless independent-variable symbols (xs, ts, ys, …).
        Any factor containing these is excluded from the coefficient.
    """
    nd_syms = nd_syms or set()
    factors = sp.Mul.make_args(term)
    param = [
        f for f in factors
        if not f.has(AppliedUndef)
        and not f.has(sp.Derivative)
        and not (nd_syms and f.free_symbols & nd_syms)
    ]
    return sp.Mul(*param) if param else sp.Integer(1)
