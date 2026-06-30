"""
Example 4: Viscous Burgers equation

  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

Scales: x ~ L, t ~ L/U, u ~ U
Non-dimensional form reveals Re = U*L/ν
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

x, t = sp.symbols("x t", positive=True)
nu, L, U = sp.symbols("nu L U", positive=True)

u = sp.Function("u")(x, t)

pde = sp.Eq(
    sp.diff(u, t) + u * sp.diff(u, x),
    nu * sp.diff(u, x, 2),
)

print("Dimensional PDE:")
print(" ", pde)
print()

nd = NonDimensionalizer(
    pde=pde,
    scales={u: U, x: L, t: L / U},
    nd_suffix="s",
    reference_term="first",
)

result = nd.run()
print(result)
