"""
Example 5: Reference value  q̄ = (q - q₀)/qc
(Langtangen & Pedersen §2.2.2 — oscillations about equilibrium)

Vertical vibration problem with gravity:
  m u'' + k u = -m g

Static equilibrium: u_eq = -mg/k.
We scale:  ū = (u - u_eq) / I  so the oscillation about equilibrium is O(1).

PDE (1D ODE written as PDE for consistency):
  m ∂²u/∂t² + k u = -m g
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

t = sp.Symbol("t", positive=True)
m, k, g, I = sp.symbols("m k g I", positive=True)

u = sp.Function("u")(t)

pde = sp.Eq(m * sp.diff(u, t, 2) + k * u, -m * g)

print("Dimensional ODE:")
print(" ", pde)
print()

u_eq = -m * g / k   # static equilibrium displacement

tc = sp.sqrt(m / k)  # natural time scale

result = NonDimensionalizer(
    pde=pde,
    scales={
        u: (I, u_eq),   # ū = (u - u_eq) / I  → oscillation amplitude I
        t: tc,
    },
    nd_suffix="s",
    reference_term="first",
).run()

print(result)
print()
print("Physical insight:")
print("  The gravity term cancels exactly when u is scaled about equilibrium.")
print("  The scaled equation is  d²ūs/dts² + ūs = 0  — no parameters remain")
print("  when initial conditions are also normalised by I.")
