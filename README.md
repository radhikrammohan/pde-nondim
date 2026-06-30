# pde-nondim

**Symbolic non-dimensionalisation of PDEs in Python.**

![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)

---

## What it does

`pde-nondim` takes a PDE written in [SymPy](https://www.sympy.org/) and user-supplied characteristic scales, and returns the fully non-dimensional form — with dimensionless groups identified and named. It also provides automatic scale discovery via dimensional analysis (Buckingham Pi) and generates ready-to-run PyTorch or JAX residuals for physics-informed neural networks (PINNs).

The package has one mandatory dependency: SymPy. No external APIs, no LaTeX parsers, no network calls.

---

## Features

| Feature | Description |
|---------|-------------|
| Correct chain-rule substitution | Walks the SymPy expression tree to transform derivatives — no broken `Subs` objects |
| Reference-value support | Scale as `q̄ = (q − q₀)/qc` for oscillations about equilibrium, temperature offsets, etc. |
| Nonlinear term handling | User-defined `f(u)`, `k(u)` get an O(1) placeholder; polynomial products handled automatically |
| Named dimensionless groups | Auto-identifies Re, Pe, Fo, Da, St, Fr, Bi, Le, Pr, Sc — forward and inverse |
| Scale suggestion | Given partial scales, finds missing scales by balancing pairs of PDE terms |
| Automatic scale discovery | `auto_scales()` enumerates all valid scale combinations via dimensional matrix / Buckingham Pi and ranks by balance score |
| Multiple scalings | Runs and compares several scale choices side-by-side |
| PINN code generation | Chainable `.to_pinn()` workflow produces complete PyTorch / JAX residual functions |
| Moving frame | `.moving_frame(xs, ts)` transforms to co-moving frame with correct chain rule |
| String input | `parse_pde()` accepts strings like `"du/dt = alpha * d2u/dx2"` |

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/your-username/pde-nondim.git
```

Editable install for development:

```bash
git clone https://github.com/your-username/pde-nondim.git
cd pde-nondim
pip install -e .
```

**Requires:** Python ≥ 3.10, SymPy ≥ 1.10

---

## Quick start

### 1. Heat equation — basic non-dimensionalisation

The convection-diffusion of heat in a moving medium:

```python
import sympy as sp
from pde_nondim import NonDimensionalizer

x, t = sp.symbols('x t', positive=True)
xs, ts = sp.symbols('xs ts', positive=True)
T = sp.Function('T')(x, t)
k, C_p, rho, V, r = sp.symbols('k C_p rho V r', positive=True)
DeltaT = sp.Symbol('DeltaT', positive=True)

pde = sp.Eq(rho * C_p * sp.diff(T, t), k * sp.diff(T, x, 2))

result = NonDimensionalizer(
    pde=pde,
    scales={T: (DeltaT, 0), x: r, t: r / V},
).run()

print(result.nd_pde_simplified)
# Eq(Derivative(Ts(xs, ts), ts) - k*Derivative(Ts(xs, ts), (xs, 2))/(C_p*V*r*rho), 0)

print(result.dimensionless_groups)
# {'1/Pe_T': k/(C_p*V*r*rho)}
```

The coefficient `k/(C_p V r ρ)` is the inverse thermal Péclet number. Setting `r = k/(C_p V ρ)` gives the diffusion length and eliminates all groups.

---

### 2. Automatic scale discovery — `auto_scales`

Given the same convection-diffusion PDE, find all valid scalings and rank them:

```python
from pde_nondim import auto_scales

dims = {
    T:   'theta',     # temperature dimension
    x:   'L',
    t:   'T',
    rho: 'M/L^3',
    C_p: 'L^2/(T^2*theta)',
    k:   'M*L/(T^3*theta)',
    V:   'L/T',
}

scalings = auto_scales(
    pde=pde,
    dims=dims,
    numerical_values={k: 7.0, C_p: 526.0, rho: 4430.0, V: 0.01, r: 0.005},
)

for s in scalings[:2]:
    print(f"Rank {s['rank']}  score={s['score']:.3f}:  {s['scales']}")

# Rank 1  score=0.000:  {T: (T0, 0), x: k/(C_p*V*rho), t: k/(C_p*V**2*rho)}
# Rank 2  score=2.121:  {T: (T0, 0), x: r, t: r/V}
```

Rank 1 is the diffusion length scale — all coefficients are O(1) and no dimensionless groups appear. Rank 2 is the convective scaling, which recovers Pe.

---

### 3. PINN workflow — `.to_pinn()` chain

For a 3-D LPBF heat equation with a Gaussian source, transform the non-dimensional PDE into a PyTorch residual function:

```python
import sympy as sp
from pde_nondim import NonDimensionalizer

# ... define pde and result as above ...
xs, ts = sp.symbols('xs ts')
Pe_sym = sp.Symbol('Pe', positive=True)
delta_sym = sp.Symbol('delta', positive=True)

pinn = (
    result.to_pinn()
    .moving_frame(xs, ts)                    # xi = xs - ts  (source fixed at xi = 0)
    .steady_state()                          # drop d/dt  (quasi-steady melt pool)
    .multiply_by(Pe_sym)                     # balance: all residual terms O(1)
    .express_as_groups({                     # replace k/(rho*Cp*V*r) -> 1/Pe, etc.
        '1/Pe_T':   sp.Integer(1) / Pe_sym,
        '1/Pe_T_2': 2*delta_sym / (sp.pi**sp.Rational(3,2) * Pe_sym),
        '1/Pe_T_3': delta_sym**2 / Pe_sym,
    })
    .set_domain(xi=(-5, 2), ys=(-3, 3), zs=(-3, 0))
    .set_parameters(Pe=(1, 60), delta=(0.1, 2.0))
)

print(pinn.pytorch_code())
```

The generated function (ready to paste into a training script):

```python
def pde_residual(xi, ys, zs, Pe, delta, model):
    # ... coordinate normalisation and autograd setup ...
    R = (
        -Pe*dTs_dxi - d2Ts_dxi2 - d2Ts_dys2 - d2Ts_dzs2*delta**2
        - 2*delta*torch.exp(-2*xi**2)*torch.exp(-2*ys**2)*torch.exp(-2*zs**2)/math.pi**(3/2)
    )
    return R
```

Use `.jax_code()` for the JAX version and `.reconstruction_code()` to recover the dimensional solution from the network output.

---

## Core API

### `NonDimensionalizer`

```python
NonDimensionalizer(pde, scales, nonlinear_scales=None, nd_suffix='s', reference_term='first')
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `pde` | `sp.Eq` or `sp.Expr` | The dimensional PDE. An `Expr` is treated as `expr == 0`. |
| `scales` | `dict` | Maps each quantity to its scale. Values may be a SymPy expression or a `(scale, ref_value)` tuple for shifted variables. |
| `nonlinear_scales` | `dict`, optional | Characteristic sizes for nonlinear sub-expressions, e.g. `{k(u): k_c}`. The expression is replaced by `k_c * ks(us)`. |
| `nd_suffix` | `str` | Suffix for dimensionless names. Default `'s'`: `x → xs`, `u → us`. |
| `reference_term` | `'first'`, `'last'`, or `int` | Which additive term's coefficient to divide through by, producing the O(1) normalised form. |

`.run()` returns a `NondimResult`:

| Field | Description |
|-------|-------------|
| `nd_pde_simplified` | The non-dimensional PDE after SymPy simplification |
| `dimensionless_groups` | `dict` mapping group name to its symbolic expression |
| `substitution_map` | `dict` mapping each original symbol to its dimensionless replacement |
| `dominant_balance` | Qualitative analysis of which terms dominate for large/small groups |

---

### `auto_scales`

```python
auto_scales(pde, dims, numerical_values=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `pde` | `sp.Eq` | The dimensional PDE. |
| `dims` | `dict` | Dimension string per symbol, e.g. `{x: 'L', rho: 'M/L^3', T: 'theta'}`. |
| `numerical_values` | `dict`, optional | Numerical values for ranking by actual coefficient magnitudes rather than symbolic balance score. |

Returns a list of dicts, each containing:

| Key | Description |
|-----|-------------|
| `'scales'` | A valid scale assignment `{symbol: (scale_expr, ref)}` |
| `'groups'` | Dimensionless groups that appear under this scaling |
| `'score'` | Balance score — 0 means all coefficients are exactly O(1); higher values indicate imbalance |
| `'rank'` | 1-based rank (1 = most balanced) |

---

### `PINNFormulation` — chainable transformations

Obtained via `result.to_pinn()`. All methods return `self` for chaining.

| Method | Description |
|--------|-------------|
| `.moving_frame(xs, ts)` | Transform to co-moving frame `ξ = xs − ts`; applies chain rule correctly |
| `.steady_state()` | Drop all time-derivative terms |
| `.multiply_by(factor)` | Multiply the residual by a scalar factor to rebalance terms |
| `.express_as_groups({name: expr})` | Replace dimensional coefficient combinations with named dimensionless groups |
| `.set_domain(**bounds)` | Set coordinate bounds, e.g. `xi=(-5, 2), ys=(-3, 3)` |
| `.set_parameters(**ranges)` | Declare parametric inputs with ranges, e.g. `Pe=(1, 60)` |
| `.pytorch_code()` | Return a string containing the complete PyTorch autograd residual function |
| `.jax_code()` | Return the JAX version of the residual |
| `.reconstruction_code()` | Return code to recover the dimensional solution from network output |

---

### `suggest_scales`

```python
suggest_scales(pde, known_scales, unknown_scales, nd_suffix='s')
```

Given known scales and a list of unknown scale symbols, finds all scale assignments that make pairs of PDE terms have equal coefficients (dominant balance).

| Parameter | Type | Description |
|-----------|------|-------------|
| `known_scales` | `dict` | Scales already decided, with unknown scale symbols included as free symbols. |
| `unknown_scales` | `list[sp.Symbol]` | Symbols to solve for. |

Returns a list of `(description_string, assignment_dict)` tuples.

---

### `MultipleScalings`

```python
MultipleScalings(pde, scale_options, labels, nd_suffix='s', reference_term='first')
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `scale_options` | `list[dict]` | One scale dict per scaling strategy. |
| `labels` | `list[str]` | Human-readable name for each strategy. |
| `reference_term` | `str`, `int`, or `list` | Single value applied to all, or one value per scaling. |

`.run_all()` → `list[(label, NondimResult)]`  
`.print_all()` → prints all results with headers

---

### `parse_pde`

```python
parse_pde(pde_str, functions, variables, parameters)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `pde_str` | `str` | PDE as a string, e.g. `"du/dt = alpha * d2u/dx2"`. |
| `functions` | `list[str]` | Dependent variable names. |
| `variables` | `list[str]` | Independent variable names. |
| `parameters` | `list[str]` | Parameter names. |

Returns `(sympy.Eq, dict_of_symbols)`.

Supported derivative notation: `du/dt`, `d2u/dx2`, `d(u)/dt`, `∂u/∂t`.

---

## Examples index

| File | PDE | Features demonstrated |
|------|-----|----------------------|
| `examples/heat_equation.py` | `∂u/∂t = α ∂²u/∂x²` | Basic usage, Fourier number |
| `examples/advection_diffusion.py` | `∂u/∂t + U∂u/∂x = D∂²u/∂x²` | Péclet number, convective vs diffusive scaling |
| `examples/burgers.py` | `∂u/∂t + u∂u/∂x = ν∂²u/∂x²` | Nonlinear, Reynolds number |
| `examples/string_input.py` | Heat equation | String parser (`parse_pde`) |
| `examples/reference_value.py` | `mu'' + ku = −mg` | Reference value, gravity cancellation |
| `examples/multiple_scalings.py` | Convection-diffusion | `MultipleScalings`, dominant balance |
| `examples/fisher_equation.py` | `∂u/∂t = α∂²u/∂x² + εu(1−u/M)` | Reaction-diffusion, parameter-free limit |
| `examples/nonlinear_diffusion.py` | `∂u/∂t = k(u)∂²u/∂x²` | Variable-coefficient diffusion, `nonlinear_scales` |
| `examples/nonlinear_reaction.py` | `∂u/∂t = α∂²u/∂x² + f(u)` | Unknown nonlinear reaction, two scalings |
| `examples/transcendental_nonlinear.py` | `∂u/∂t = α∂²u/∂x² + A exp(−E/u)` | Arrhenius term, automatic vs explicit O(1) |
| `examples/lpbf_gaussian.py` | LPBF 2-D Gaussian heat source | Rosenthal scaling, Péclet number |
| `examples/lpbf_3d_gaussian.py` | LPBF 3-D Gaussian with depth scale | Beam aspect ratio δ = r/d |
| `examples/lpbf_large_q0.py` | LPBF with phase change | Stefan number, melting threshold |
| `examples/lpbf_parametric_study.py` | LPBF over (P, V) sweep | Parametric study strategy, reconstruction |
| `examples/lpbf_pinn_formulation.py` | LPBF full PINN pipeline | `to_pinn()`, moving frame, PyTorch code |

Run any example from the repo root:

```bash
PYTHONPATH=. python3 examples/heat_equation.py
```

Or after `pip install -e .`:

```bash
python3 examples/heat_equation.py
```

---

## Recognised dimensionless groups

| Symbol | Name | Physical ratio |
|--------|------|----------------|
| `Re` | Reynolds | inertia / viscous |
| `Pe` | Péclet (mass) | convection / diffusion |
| `Pe_T` | Péclet (heat) | convection / thermal diffusion |
| `Fo` | Fourier | diffusion time / imposed time |
| `Da` | Damköhler II | reaction / diffusion |
| `Da_I` | Damköhler I | reaction / convection |
| `St` | Strouhal | oscillation / convection |
| `Fr` | Froude | inertia / gravity |
| `Bi` | Biot | convective / conductive heat transfer |
| `Le` | Lewis | thermal diffusivity / mass diffusivity |
| `Pr` | Prandtl | momentum diffusivity / thermal diffusivity |
| `Sc` | Schmidt | momentum diffusivity / mass diffusivity |

Both a group (`Pe`) and its inverse (`1/Pe`) are detected. Unknown groups are labelled `Pi_1`, `Pi_2`, …

---

## Methodology

The implementation follows the five-step procedure from Langtangen & Pedersen (2016):

1. Identify independent and dependent variables.
2. Introduce dimensionless variables `q̄ = (q − q₀)/qc`.
3. Substitute using the chain rule: `∂ⁿu/∂xⁿ → (U/Lⁿ) ∂ⁿū/∂x̄ⁿ`.
4. Normalise by dividing through by one term's coefficient to obtain the O(1) form.
5. Interpret the remaining coefficients as dimensionless groups.

Scale selection follows the dominant-balance approach: scales are chosen so that the terms of greatest physical interest have coefficient 1, and the remaining groups express ratios of competing effects. The `auto_scales` function enumerates all dimensionally consistent scale combinations via the dimensional matrix and Buckingham Pi theorem, then ranks them by how close the resulting coefficients are to unity.

> Langtangen, H.P. & Pedersen, G.K., *Scaling of Differential Equations*, Springer (2016).

---

## Known limitations

**Supported nonlinear forms**

The package handles a wide range of nonlinear PDEs without any special configuration:

- Polynomial coefficients: `u^n u_xx`, `u * u_x` (Burgers), `u^3 - u` (Allen-Cahn)
- Divergence form with nonlinear diffusivity: `∂/∂x(k(u) ∂u/∂x)` — both the `k(u)u_xx` and `k'(u)u_x²` terms are scaled correctly
- Powers of user-defined functions: `c(u)² u_xx`
- Higher-order PDEs: `∂⁴u/∂x⁴` (Cahn-Hilliard, beam equations)
- Transcendental source terms: `A exp(-E/u)` (Arrhenius)

For any term where the characteristic size cannot be inferred from the PDE structure alone, pass it explicitly via `nonlinear_scales`:

```python
result = NonDimensionalizer(
    pde,
    scales={u: U, x: L, t: T},
    nonlinear_scales={
        k(u): k_c,          # k(u) -> k_c * ks(us)
        theta(h): theta_c,  # theta(h) -> theta_c * thetas(hs)
    },
)
```

**Current limitations**

- **Single PDE only** — coupled PDE systems (e.g. Navier-Stokes velocity-pressure, reaction-diffusion with two species) are not supported. Each equation must be non-dimensionalised separately, and the user is responsible for ensuring consistent scales across equations.
- **Single dependent variable** — cross-products of two different dependent variables (e.g. `u·v` in a coupled system) are not handled. This is a direct consequence of the single-PDE limitation above.
- **p-Laplacian and gradient-magnitude coefficients** — terms like `|∇u|^(p-2) ∇u` require an explicit `nonlinear_scales` entry for the gradient term.
- **Heuristic group naming** — group identification works reliably for standard symbol names (`nu`, `alpha`, `L`, `U`, `D`, `rho`, `Cp`). Unconventional naming may produce `Pi_1`, `Pi_2`, … instead of a named group.
- **No LaTeX input** — SymPy is the required input format. This is intentional: SymPy expressions are unambiguous, version-controllable, and require no external parser or API.

---

## Contributing

Bug reports and pull requests are welcome. Please open an issue before starting significant work so we can discuss scope and approach. The test suite runs with `pytest`:

```bash
pytest tests/
```

Code should be compatible with Python 3.10+ and pass `pytest` with no external dependencies beyond SymPy.

---

## License

MIT
