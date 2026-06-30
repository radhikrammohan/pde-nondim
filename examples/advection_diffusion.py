"""
Example 2: Advection-diffusion equation

  ∂u/∂t + U₀ ∂u/∂x = D ∂²u/∂x²

Characteristic scales:
  x ~ L, t ~ L/U₀, u ~ C₀

Non-dimensional form reveals the Péclet number Pe = U₀ L / D
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

x, t = sp.symbols("x t", positive=True)
U0, D, L, C0 = sp.symbols("U0 D L C0", positive=True)

u = sp.Function("u")(x, t)

pde = sp.Eq(
    sp.diff(u, t) + U0 * sp.diff(u, x),
    D * sp.diff(u, x, 2),
)

print("Dimensional PDE:")
print(" ", pde)
print()

nd = NonDimensionalizer(
    pde=pde,
    scales={
        u: C0,
        x: L,
        t: L / U0,   # convective time scale
    },
    nd_suffix="s",
    reference_term=1,  # normalise by advection term (index 1)
)

result = nd.run()
print(result)
