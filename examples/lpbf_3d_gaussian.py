"""
Example 13: LPBF — 3-D Gaussian heat source with depth dependence

Governing PDE — full 3-D transient heat conduction:

  rho*Cp * dT/dt = k*(d2T/dx2 + d2T/dy2 + d2T/dz2) + Q(x,y,z,t)

3-D Gaussian volumetric heat source:

  Q(x,y,z,t) = Q0 * exp( -2*(x-Vt)^2/r^2 - 2*y^2/r^2 - 2*z^2/d^2 )

  r = beam radius     (lateral half-width at 1/e^2)  [m]
  d = depth parameter (penetration depth)             [m]

  Q0 [W/m^3] = peak volumetric heat generation
     For 3-D Gaussian: Q0 = 2*P*eta / (pi^(3/2) * r^2 * d)

Depth dependence:
  d = r             -> spherical Gaussian (isotropic, conduction mode)
  d > r             -> deep penetration  (keyhole mode, laser drills deep)
  d < r             -> shallow heating   (surface conduction only)

Non-dimensionalisation
----------------------
Two natural length scales exist:
  r  ->  xs = x/r,  ys = y/r     (lateral, set by beam radius)
  d  ->  zs = z/d                 (depth,   set by penetration depth)
  tc = r/V                        (convective time, beam transit)
  T = T0 + (Q0*r^2/k)*Ts         (Rosenthal-type temperature scale)

Substituting gives the canonical form:

  Pe * dTs/dts = d2Ts/dxs2 + d2Ts/dys2
               + delta^2 * d2Ts/dzs2
               + exp( -2*(xs-ts)^2 - 2*ys^2 - 2*zs^2 )

Two dimensionless groups:
  Pe    = rho*Cp*V*r/k   (thermal Peclet — scan speed vs diffusion)
  delta = r/d             (aspect ratio   — beam width vs melt depth)

Physical regimes:
  Pe >> 1, delta ~ 1   ->  convection-dominated, near-spherical melt pool
  Pe >> 1, delta >> 1  ->  keyhole mode, narrow deep melt pool
  Pe << 1              ->  quasi-static melt pool (low scan speed)
  delta -> 0           ->  2-D limit (no depth variation, d -> infinity)
"""
import sympy as sp
from pde_nondim import NonDimensionalizer

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------
x, y, z, t = sp.symbols("x y z t", positive=True)
rho, Cp, k, Q0, r, d, V, T0 = sp.symbols(
    "rho C_p k Q_0 r d V T_0", positive=True
)

T = sp.Function("T")(x, y, z, t)

# 3-D Gaussian: lateral scale r, depth scale d
pde = sp.Eq(
    rho * Cp * sp.diff(T, t),
    k * (sp.diff(T, x, 2) + sp.diff(T, y, 2) + sp.diff(T, z, 2))
    + Q0 * sp.exp(-2*(x - V*t)**2/r**2 - 2*y**2/r**2 - 2*z**2/d**2),
)

print("=" * 65)
print("LPBF 3-D HEAT EQUATION WITH DEPTH-DEPENDENT GAUSSIAN SOURCE")
print("=" * 65)
print("\nDimensional PDE:")
print(" ", pde)
print()

# -----------------------------------------------------------------------
# Non-dimensionalise
# Different length scales in lateral (r) and depth (d) directions
# -----------------------------------------------------------------------
Tc = Q0 * r**2 / k       # characteristic temperature

result = NonDimensionalizer(
    pde,
    scales={
        T: (Tc, T0),      # T = T0 + (Q0*r^2/k) * Ts
        x: r,             # xs = x/r
        y: r,             # ys = y/r
        z: d,             # zs = z/d   <- different scale!
        t: r / V,         # ts = Vt/r
    },
).run()

print(result)

# -----------------------------------------------------------------------
# Interpret the two groups
# -----------------------------------------------------------------------
print("\nGroup interpretation")
print("-" * 65)
print("The non-dimensional PDE has exactly TWO parameters:\n")
print("  Pe = rho*Cp*V*r/k    (thermal Peclet number)")
print("     = (convective transport) / (lateral diffusion)")
print()
print("  delta^2 = (r/d)^2   appears as coefficient of d2Ts/dzs2")
print("     delta = r/d = beam_radius / penetration_depth")
print()
print("Full dimensionless form:")
print("  Pe*dTs/dts = d2Ts/dxs2 + d2Ts/dys2")
print("             + delta^2 * d2Ts/dzs2")
print("             + exp(-2*(xs-ts)^2 - 2*ys^2 - 2*zs^2)")
print()
print("  delta  = 1    ->  isotropic diffusion (spherical melt pool)")
print("  delta >> 1   ->  depth diffusion amplified (keyhole, deep narrow pool)")
print("  delta << 1   ->  depth diffusion suppressed (shallow wide pool)")

# -----------------------------------------------------------------------
# Numerical checks for three LPBF modes
# -----------------------------------------------------------------------
import math

rho_v = 4430   # kg/m^3  Ti-6Al-4V
Cp_v  = 560    # J/kgK
k_v   = 7.0    # W/mK
P_v   = 200    # W
eta_v = 0.35

print()
print("=" * 65)
print("NUMERICAL CHECK — Three LPBF operating modes (Ti-6Al-4V)")
print("=" * 65)

modes = [
    ("Conduction mode  (r=d)",          1.0,  50e-6,  50e-6),
    ("Transition mode  (d = 2r)",        1.0,  50e-6, 100e-6),
    ("Keyhole mode     (d = 5r)",        2.0,  50e-6, 250e-6),
]

for label, V_v, r_v, d_v in modes:
    Q0_v   = 2*P_v*eta_v / (math.pi**1.5 * r_v**2 * d_v)
    Pe_v   = rho_v * Cp_v * V_v * r_v / k_v
    delta_v = r_v / d_v
    Tc_v   = Q0_v * r_v**2 / k_v

    print(f"\n  {label}")
    print(f"    V={V_v} m/s, r={r_v*1e6:.0f} um, d={d_v*1e6:.0f} um")
    print(f"    Pe     = {Pe_v:.2f}")
    print(f"    delta  = r/d = {delta_v:.2f}")
    print(f"    delta^2 = {delta_v**2:.2f}  (depth diffusion coefficient in dim'less PDE)")
    print(f"    1/Pe   = {1/Pe_v:.4f}  (lateral diffusion coefficient)")
    print(f"    Tc     = {Tc_v:.0f} K  (characteristic temp rise)")

    vals = {rho: rho_v, Cp: Cp_v, k: k_v, Q0: Q0_v,
            r: r_v, d: d_v, V: V_v, T0: 300}
    print()
    print(result.check_magnitudes(vals))

# -----------------------------------------------------------------------
# How delta changes the melt pool shape
# -----------------------------------------------------------------------
print()
print("Melt pool geometry (qualitative — from delta = r/d):")
print(f"  {'d/r':>8}  {'delta':>8}  {'depth_coeff (delta^2)':>22}  Pool shape")
print("  " + "-" * 60)
for d_over_r in [0.5, 1.0, 2.0, 5.0, 10.0]:
    delta   = 1.0 / d_over_r
    delta_sq = delta**2
    shape = ("elongated deep (keyhole)" if d_over_r >= 5
             else "near-spherical" if d_over_r == 1
             else "shallow wide (conduction)" if d_over_r <= 0.5
             else "ellipsoidal")
    print(f"  {d_over_r:>8.1f}  {delta:>8.3f}  {delta_sq:>22.3f}  {shape}")
