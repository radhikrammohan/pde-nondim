"""
Example 11: Numerical magnitude check

Shows how check_magnitudes() evaluates each dimensionless group
for a specific physical regime and flags terms that are not O(1).

Three regimes of the advection-diffusion equation are tested:
  A. Pe ~ 1   -> well-balanced, all terms O(1)
  B. Pe >> 1  -> convection dominates, diffusion negligible
  C. Pe << 1  -> diffusion dominates, convection negligible
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

x, t = sp.symbols("x t", positive=True)
V, D, L, C0 = sp.symbols("V D L C0", positive=True)
u = sp.Function("u")(x, t)

pde = sp.Eq(sp.diff(u, t) + V * sp.diff(u, x), D * sp.diff(u, x, 2))

# Scale with convective time tc = L/V
result = NonDimensionalizer(pde, scales={u: C0, x: L, t: L / V}).run()

print(result)
print()

# ------------------------------------------------------------------
# Regime A: Pe ~ 1  (V=0.1, L=1, D=0.1 -> Pe = V*L/D = 1)
# ------------------------------------------------------------------
print("REGIME A  —  Pe = 1  (balanced)")
print(result.check_magnitudes({V: 0.1, L: 1.0, D: 0.1, C0: 1.0}))

# ------------------------------------------------------------------
# Regime B: Pe >> 1  (V=10, L=1, D=0.01 -> Pe = 1000)
# ------------------------------------------------------------------
print()
print("REGIME B  —  Pe = 1000  (convection-dominated)")
print(result.check_magnitudes({V: 10.0, L: 1.0, D: 0.01, C0: 1.0}))

# ------------------------------------------------------------------
# Regime C: Pe << 1  (V=0.001, L=1, D=1 -> Pe = 0.001)
# ------------------------------------------------------------------
print()
print("REGIME C  —  Pe = 0.001  (diffusion-dominated)")
print(result.check_magnitudes({V: 0.001, L: 1.0, D: 1.0, C0: 1.0}))

# ------------------------------------------------------------------
# Burgers at high Reynolds number
# ------------------------------------------------------------------
print()
print("=" * 65)
print("Burgers equation at Re = 10000  (turbulent-like regime)")
print("=" * 65)
nu_sym, U_sym = sp.symbols("nu U", positive=True)
u2 = sp.Function("u")(x, t)
burgers = sp.Eq(
    sp.diff(u2, t) + u2 * sp.diff(u2, x),
    nu_sym * sp.diff(u2, x, 2),
)
r2 = NonDimensionalizer(burgers, scales={u2: U_sym, x: L, t: L / U_sym}).run()
print(r2.check_magnitudes({nu_sym: 1e-4, U_sym: 1.0, L: 1.0}))
print()
print("At Re=10000, viscous term is O(1/Re)=O(0.0001) — negligible.")
print("Scale with t_c = L/U and the viscous term drops out at leading order.")
