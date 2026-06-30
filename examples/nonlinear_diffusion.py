"""
Example 8: Nonlinear diffusion with variable coefficient  k(u)

Two common forms appear in practice:

Form A — divergence form (heat/mass transport with variable conductivity):
  du/dt = d/dx[ k(u) * du/dx ]
  Expanded: du/dt = k(u)*d2u/dx2 + k'(u)*(du/dx)^2

Form B — simplified (when k'(u) terms are small, or k depends on x not u):
  du/dt = k(u) * d2u/dx2

We demonstrate Form B here, which is the cleaner case for non-dimensionalisation.
For Form A, expand manually and provide nonlinear_scales for both k(u) and k'(u).

Scales: x ~ L, t ~ T, u ~ U, k(u) ~ k_c
"""
import sympy as sp
from pde_nondim import NonDimensionalizer, MultipleScalings

x, t = sp.symbols("x t", positive=True)
L, T, U, k_c = sp.symbols("L T U k_c", positive=True)

u = sp.Function("u")(x, t)
k = sp.Function("k")       # unknown nonlinear conductivity

# Form B: du/dt = k(u) * d2u/dx2
pde = sp.Eq(
    sp.diff(u, t),
    k(u) * sp.diff(u, x, 2),
)

print("Dimensional PDE (Form B):")
print(" ", pde)
print()

# Scales: k(u) has characteristic size k_c
# -> k(u) = k_c * ks(us)  where ks is a dimensionless function of us
result = NonDimensionalizer(
    pde=pde,
    scales={u: U, x: L, t: T},
    nonlinear_scales={k(u): k_c},
    nd_suffix="s",
    reference_term="first",
).run()

print(result)

# -----------------------------------------------------------------------
# Physical insight: two natural time scales
# -----------------------------------------------------------------------
print()
print("Scale suggestion — what time scale balances the diffusion term?")
from pde_nondim import suggest_scales

tc = sp.Symbol("tc", positive=True)
candidates = suggest_scales(
    pde=pde,
    known_scales={u: U, x: L, t: tc},
    unknown_scales=[tc],
)
for desc, val in candidates:
    print(" ", desc)

print()
print("Natural time scale: tc = L^2 / k_c  (diffusive, using k_c as effective diffusivity)")
print()

result2 = NonDimensionalizer(
    pde=pde,
    scales={u: U, x: L, t: L**2 / k_c},
    nonlinear_scales={k(u): k_c},
    nd_suffix="s",
    reference_term="first",
).run()

print("With tc = L^2/k_c:")
print(result2)
print("-> PDE has no parameters: du_s/dt_s = k_s(u_s) * d2u_s/dx_s2")
print("   All physics is in the shape of k_s(u_s), not separate parameters.")
