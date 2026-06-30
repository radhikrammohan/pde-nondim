"""
Example 7: Fisher / KPP reaction-diffusion equation
(Langtangen & Pedersen §3.3.1)

  ∂u/∂t = α ∂²u/∂x² + ε u(1 - u/M)

Two scalings:
  A. "All terms balance" — choose tc = 1/ε, xc = sqrt(α/ε), uc = M
     → PDE with NO parameters (§3.3.1 eq. 3.61)

  B. Fixed domain length L — choose tc = 1/ε, xc = L, uc = M
     → introduces β = α/(ε L²)  (§3.3.1 eq. 3.62)
"""
import sympy as sp
from pde_nondim import MultipleScalings

x, t = sp.symbols("x t", positive=True)
alpha, eps, M, L = sp.symbols("alpha epsilon M L", positive=True)

u = sp.Function("u")(x, t)

pde = sp.Eq(
    sp.diff(u, t),
    alpha * sp.diff(u, x, 2) + eps * u * (1 - u / M),
)

print("Dimensional PDE (Fisher/KPP):")
print(" ", pde)
print()

ms = MultipleScalings(
    pde=pde,
    scale_options=[
        # A: all-terms balance — unique scales with no parameters
        {u: M, x: sp.sqrt(alpha / eps), t: 1 / eps},
        # B: fixed domain length L — introduces β = α/(ε L²)
        {u: M, x: L, t: 1 / eps},
    ],
    labels=[
        "All-terms balance  xc=√(α/ε), tc=1/ε  → parameter-free PDE",
        "Fixed domain length  xc=L, tc=1/ε      → Damköhler group appears",
    ],
    reference_term="first",
)
ms.print_all()
