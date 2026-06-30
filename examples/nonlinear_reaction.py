"""
Example 9: General nonlinear reaction-diffusion  (Langtangen §3.3.2)

  du/dt = alpha * d2u/dx2 + f(u)

where f(u) is a general nonlinear reaction term with characteristic
size f_c = max|f(u)|.

The book shows two natural scale choices:
  A. Reaction time scale:  tc = U/f_c  -> diffusion group beta appears
  B. Diffusion time scale: tc = L^2/alpha -> 1/beta in front of f

This reveals beta = alpha*U / (L^2 * f_c) = ratio of reaction time to
diffusion time.
"""
import sympy as sp
from pde_nondim import NonDimensionalizer, MultipleScalings

x, t = sp.symbols("x t", positive=True)
alpha, L, U, f_c = sp.symbols("alpha L U f_c", positive=True)

u = sp.Function("u")(x, t)
f = sp.Function("f")    # unknown nonlinear reaction

pde = sp.Eq(
    sp.diff(u, t),
    alpha * sp.diff(u, x, 2) + f(u),
)

print("Dimensional PDE:")
print(" ", pde)
print()

# Tell the code: f(u) has characteristic size f_c
nl = {f(u): f_c}

ms = MultipleScalings(
    pde=pde,
    nonlinear_scales_options=nl,   # applied to both scalings
    scale_options=[
        # A: reaction time scale tc = U/f_c
        {u: U, x: L, t: U / f_c},
        # B: diffusion time scale tc = L^2/alpha
        {u: U, x: L, t: L**2 / alpha},
    ],
    labels=[
        "Reaction time scale  tc = U/f_c   -> diffusion group (beta) appears",
        "Diffusion time scale tc = L^2/a   -> 1/beta in front of reaction",
    ],
    reference_term=["first", "first"],
)
ms.print_all()

print()
print("beta = alpha*U/(L^2 * f_c) = reaction_timescale / diffusion_timescale")
print("  beta >> 1 : diffusion dominates -> du/dt ~ alpha*d2u/dx2")
print("  beta << 1 : reaction dominates  -> du/dt ~ f(u)")
