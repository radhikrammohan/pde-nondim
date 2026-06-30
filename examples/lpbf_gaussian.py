"""
Example 12: LPBF (Laser Powder Bed Fusion) with Gaussian heat source

Governing PDE — 2-D transient heat conduction (x-y plane):

  rho * Cp * dT/dt = k * (d2T/dx2 + d2T/dy2) + Q(x, y, t)

Gaussian volumetric heat source (laser scanning at speed V in x):

  Q(x, y, t) = Q0 * exp( -2*((x - V*t)^2 + y^2) / r^2 )

Parameters
----------
  rho   : density                   [kg/m^3]
  Cp    : specific heat             [J/(kg.K)]
  k     : thermal conductivity      [W/(m.K)]
  Q0    : peak heat source          [W/m^3]    <- volumetric intensity
  r     : beam radius (1/e^2)       [m]
  V     : scan speed                [m/s]
  T0    : ambient temperature       [K]

  For a 3-D spherical Gaussian:  Q0 = 2*P*eta / (pi^(3/2) * r^3)
  For a surface flux model:      Q0 = 2*P*eta / (pi * r^2 * h)   (h = layer thickness)

Non-dimensionalisation
----------------------
  T = T0 + Tc * Ts(xs, ys, ts)         Tc = Q0 * r^2 / k
  x = r * xs,  y = r * ys
  t = (r/V) * ts                        (convective time, beam transit time)

Expected non-dimensional form:
  Pe * dTs/dts = d2Ts/dxs2 + d2Ts/dys2 + exp(-2*(xs - ts)^2 - 2*ys^2)

  Pe = rho * Cp * V * r / k             (thermal Peclet number)

Physical regimes:
  Pe >> 1  : convection-dominated  (fast scan / low diffusivity)
             -> thin elongated melt pool, steep thermal gradients
  Pe << 1  : diffusion-dominated   (slow scan / high diffusivity)
             -> wide deep melt pool, quasi-static solution
  Pe ~ 1   : both equally important -> transition regime
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

# -----------------------------------------------------------------------
# Symbols
# -----------------------------------------------------------------------
x, y, t = sp.symbols("x y t", positive=True)
rho, Cp, k, Q0, r, V, T0 = sp.symbols("rho C_p k Q_0 r V T_0", positive=True)

T = sp.Function("T")(x, y, t)

# 2-D transient heat equation with moving Gaussian source
pde = sp.Eq(
    rho * Cp * sp.diff(T, t),
    k * (sp.diff(T, x, 2) + sp.diff(T, y, 2))
    + Q0 * sp.exp(-2 * ((x - V * t)**2 + y**2) / r**2),
)

print("=" * 65)
print("LPBF HEAT EQUATION WITH GAUSSIAN SOURCE")
print("=" * 65)
print("\nDimensional PDE:")
print(" ", pde)
print()

# -----------------------------------------------------------------------
# Non-dimensionalise
# Characteristic temperature: Tc = Q0 * r^2 / k
# Convective time scale:      tc = r / V
# -----------------------------------------------------------------------
Tc = Q0 * r**2 / k

result = NonDimensionalizer(
    pde,
    scales={
        T: (Tc, T0),     # T = T0 + (Q0*r^2/k) * Ts(xs, ys, ts)
        x: r,            # xs = x/r
        y: r,            # ys = y/r
        t: r / V,        # ts = V*t/r   (beam transit time)
    },
).run()

print(result)

# -----------------------------------------------------------------------
# Interpretation
# -----------------------------------------------------------------------
print("\nPhysical interpretation")
print("-" * 65)
print("Substituting T = T0 + (Q0*r^2/k)*Ts, xs=x/r, ys=y/r, ts=Vt/r:")
print()
print("  Pe * dTs/dts = d2Ts/dxs2 + d2Ts/dys2")
print("               + exp( -2*(xs - ts)^2 - 2*ys^2 )")
print()
print("  -> ONE parameter: Pe = rho*Cp*V*r/k  (thermal Peclet number)")
print()
print("  The Gaussian is fixed at O(1) — all the physics is in Pe.")
print("  Coefficient of diffusion & source terms = 1/Pe.")
print("  Coefficient of convective (time) term   = 1   (reference).")
print()
print("Regimes:")
print("  Pe >> 1  convection-dominated: thin melt pool, steep gradients")
print("  Pe ~  1  transition:           diffusion and convection balanced")
print("  Pe << 1  diffusion-dominated:  wide melt pool, quasi-static")

# -----------------------------------------------------------------------
# Numerical check — Ti-6Al-4V typical LPBF parameters
# -----------------------------------------------------------------------
print()
print("=" * 65)
print("NUMERICAL CHECK — Ti-6Al-4V")
print("=" * 65)

import math

P_val   = 200       # W
eta_val = 0.35      # absorptivity
r_val   = 50e-6     # m  (50 micron beam radius)
V_val   = 1.0       # m/s
rho_val = 4430      # kg/m^3
Cp_val  = 560       # J/(kg.K)
k_val   = 7.0       # W/(m.K)
T0_val  = 300       # K

# Volumetric intensity (spherical 3-D Gaussian)
Q0_val = 2 * P_val * eta_val / (math.pi**1.5 * r_val**3)
Pe_val  = rho_val * Cp_val * V_val * r_val / k_val
Tc_val  = Q0_val * r_val**2 / k_val

print(f"\n  Material: Ti-6Al-4V")
print(f"  rho={rho_val} kg/m3, Cp={Cp_val} J/kgK, k={k_val} W/mK")
print(f"  P={P_val} W, eta={eta_val}, r={r_val*1e6:.0f} um, V={V_val} m/s")
print(f"\n  Q0  = 2*P*eta/(pi^1.5*r^3) = {Q0_val:.2e} W/m^3")
print(f"  Tc  = Q0*r^2/k             = {Tc_val:.0f} K")
print(f"  Pe  = rho*Cp*V*r/k         = {Pe_val:.1f}")
print(f"\n  -> Convection-dominated (Pe={Pe_val:.1f} >> 1)")
print(f"     1/Pe = {1/Pe_val:.3f}  (diffusion coefficient in dimensionless PDE)")

ti64 = {rho: rho_val, Cp: Cp_val, k: k_val,
        Q0: Q0_val, r: r_val, V: V_val, T0: T0_val}

print()
print(result.check_magnitudes(ti64))

# -----------------------------------------------------------------------
# Scan speed sensitivity
# -----------------------------------------------------------------------
print()
print("Scan speed sensitivity:")
print(f"  {'V (m/s)':>10}  {'Pe':>8}  {'1/Pe':>8}  Regime")
print("  " + "-" * 50)
for V_i in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
    Pe = rho_val * Cp_val * V_i * r_val / k_val
    regime = ("diffusion-dominated" if Pe < 0.1
              else "convection-dominated" if Pe > 10
              else "transition")
    print(f"  {V_i:>10.2f}  {Pe:>8.2f}  {1/Pe:>8.4f}  {regime}")
