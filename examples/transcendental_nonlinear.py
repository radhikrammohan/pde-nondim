"""
Example 10: Transcendental nonlinearity  exp(u)

Arrhenius-type reaction-diffusion (combustion, chemical kinetics):

  du/dt = alpha * d2u/dx2 + A * exp(-E/u)

where A is the pre-exponential factor and E is an activation energy.
The reaction term exp(-E/u) is transcendental in u.

Characteristic size of exp(-E/u) evaluated at u=U:  exp(-E/U)

Three approaches:
  1. No nonlinear_scales  -> argument gets substituted automatically
  2. With nonlinear_scales -> exp(-E/u) is normalised to exactly O(1)
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

x, t = sp.symbols("x t", positive=True)
alpha, L, T, U, A, E = sp.symbols("alpha L T U A E", positive=True)

u = sp.Function("u")(x, t)

# Arrhenius reaction term
reaction = A * sp.exp(-E / u)

pde = sp.Eq(
    sp.diff(u, t),
    alpha * sp.diff(u, x, 2) + reaction,
)

print("Dimensional PDE:")
print(" ", pde)
print()

# -----------------------------------------------------------------------
# Approach 1: No nonlinear_scales — argument is substituted, no O(1) norm
# -----------------------------------------------------------------------
print("=" * 60)
print("Approach 1: automatic argument substitution (no explicit nl scale)")
print("=" * 60)
result1 = NonDimensionalizer(
    pde=pde,
    scales={u: U, x: L, t: T},
    nd_suffix="s",
).run()
print(result1)

# -----------------------------------------------------------------------
# Approach 2: Provide exp(-E/u) scale -> O(1) normalisation
# The characteristic size of exp(-E/u) at u=U is exp(-E/U)
# -----------------------------------------------------------------------
print()
print("=" * 60)
print("Approach 2: explicit nonlinear scale -> O(1) transcendental")
print("=" * 60)
result2 = NonDimensionalizer(
    pde=pde,
    scales={u: U, x: L, t: T},
    nonlinear_scales={sp.exp(-E / u): sp.exp(-E / U)},
    nd_suffix="s",
).run()
print(result2)
print()
print("The exps(us) function in approach 2 is O(1) by construction.")
print("Its coefficient A*T*exp(-E/U) is the Damkoehler-type group.")
