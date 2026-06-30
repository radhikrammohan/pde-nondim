"""
Example 1: Heat equation (diffusion)

  ∂u/∂t = α ∂²u/∂x²

Characteristic scales:
  x ~ L  (domain length)
  t ~ T  (time scale — left as free so we can show the Fourier number)
  u ~ ΔT (temperature difference)
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

# --- Dimensional setup ---
x, t = sp.symbols("x t", positive=True)
alpha, L, T_scale, Delta_T = sp.symbols("alpha L T Delta_T", positive=True)

u = sp.Function("u")(x, t)

pde = sp.Eq(sp.diff(u, t), alpha * sp.diff(u, x, 2))

print("Dimensional PDE:")
print(" ", pde)
print()

# --- Non-dimensionalise ---
nd = NonDimensionalizer(
    pde=pde,
    scales={
        u: Delta_T,
        x: L,
        t: T_scale,
    },
    nd_suffix="s",          # u → us, x → xs, t → ts
    reference_term="first", # normalise by the time-derivative coefficient
)

result = nd.run()
print(result)
