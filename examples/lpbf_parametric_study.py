"""
Example 15: LPBF parametric study — how to non-dimensionalise for a
range of inputs (P, V, domain size) with temperature as output.

GOAL
----
Given inputs:
  - Laser power     P  [W]
  - Scan speed      V  [m/s]
  - Beam radius     r  [m]
  - Depth parameter d  [m]
  - Domain          Lx, Ly, Lz, t_total
  - Material        rho, Cp, k

Predict output:
  - Temperature field  T(x, y, z, t)

PROBLEM
-------
Running a full simulation for every (P, V) combination is expensive.
Non-dimensionalisation reveals which combinations give the SAME physics,
so a single dimensionless solution maps to many dimensional cases.

BUCKINGHAM PI ANALYSIS
----------------------
Independent dimensional variables: P, V, r, d, k, rhoCP (= rho*Cp)
4 fundamental dimensions: [W, m, K, s]
→ 6 - 4 = 2 free dimensionless parameters control the PDE shape:

  Pe = rho*Cp*V*r / k        (thermal Peclet — scan speed vs diffusion)
  delta = r / d              (beam aspect ratio — lateral vs depth)

CRITICAL RESULT
---------------
Laser power P appears ONLY in the temperature scale:

  Tc = P * eta / (k * r)     [K]

So P never changes the SHAPE of the temperature field — only its
magnitude. Two cases with different P but same Pe give the same
dimensionless solution; dimensional T just scales linearly with P.

UNIVERSAL SOLUTION
------------------
The dimensionless PDE (normalised by convective coefficient):

  dTs/dts = (1/Pe) * [d2Ts/dxs2 + d2Ts/dys2]
           + (delta^2/Pe) * d2Ts/dzs2
           + (2*delta/(pi^(3/2)*Pe)) * exp(-2(xs-ts)^2 - 2ys^2 - 2zs^2)

  subject to:
    Ts = 0  at  t=0  (preheat = T0, reference absorbed into Tc)
    domain:  xs in [0, Lx/r],  ys in [-Ly/(2r), Ly/(2r)],
             zs in [-Lz/d, 0],  ts in [0, V*t_total/r]

  Ts depends on: Pe, delta, and domain aspect ratios Lx/r, Ly/r, Lz/d

RECONSTRUCTION
--------------
For any dimensional case (P, V, r, d, material):

  1. Compute  Pe    = rho*Cp*V*r / k
              delta = r / d
              Tc    = P*eta / (k*r)

  2. Look up (or interpolate) the dimensionless solution
     Ts(xs, ys, zs, ts;  Pe, delta)

  3. Reconstruct:
     T(x,y,z,t) = T0 + Tc * Ts(x/r, y/r, z/d, V*t/r)
                = T0 + [P*eta/(k*r)] * Ts(...)

PARAMETRIC STUDY STRATEGY
--------------------------
Instead of running N_P * N_V simulations, run N_Pe * N_delta simulations:

  Typical ranges for LPBF (Ti-6Al-4V, r=50um):
    P  in [50, 400] W      -> Tc in [~5000, ~40000] K  (just a scale)
    V  in [0.1, 3.0] m/s  -> Pe in [1.8, 53]
    d  in [r, 5r]          -> delta in [0.2, 1.0]

  So 5 Pe values x 4 delta values = 20 simulations cover the space
  vs 8 P values x 10 V values = 80 dimensional simulations.

  Factor-of-4 reduction here; much larger for wider ranges.

DOMAIN AS DIMENSIONLESS BOUNDARY CONDITIONS
--------------------------------------------
Domain size appears only in the domain bounds, NOT in the PDE:

  Physical domain   Lx x Ly x Lz x t_total
  Dimensionless     (Lx/r) x (Ly/r) x (Lz/d) x (V*t_total/r)

  For semi-infinite (large domain):  Lx/r, Ly/r, Lz/d >> 1
  -> boundary effects negligible, universal Rosenthal solution applies

  For finite domain: need Lx/r, Ly/r, Lz/d as additional parameters.
  Rule of thumb: if domain > 10r in each direction, semi-infinite OK.
"""
import math
import numpy as np

# -----------------------------------------------------------------------
# Scaling functions
# -----------------------------------------------------------------------

def compute_groups(P, eta, V, r, d, rho, Cp, k, T0=300):
    """Return (Pe, delta, Tc) for a set of dimensional inputs."""
    Pe    = rho * Cp * V * r / k
    delta = r / d
    Tc    = P * eta / (k * r)
    return Pe, delta, Tc


def dimensionless_domain(Lx, Ly, Lz, t_total, r, d, V):
    """Convert physical domain to dimensionless extents."""
    return {
        "xs_max": Lx / r,
        "ys_half": Ly / (2 * r),
        "zs_depth": Lz / d,
        "ts_max": V * t_total / r,
    }


def reconstruct_T(Ts, Tc, T0):
    """Recover dimensional temperature from dimensionless solution."""
    return T0 + Tc * Ts


# -----------------------------------------------------------------------
# Print the non-dimensionalisation table for a parameter sweep
# -----------------------------------------------------------------------

print("=" * 70)
print("LPBF PARAMETRIC STUDY — Ti-6Al-4V")
print("=" * 70)

rho = 4430   # kg/m^3
Cp  = 560    # J/(kg.K)
k   = 7.0    # W/(m.K)
eta = 0.35
r   = 50e-6  # m
d   = 50e-6  # m   (conduction mode, d = r)
T0  = 300    # K

print(f"\nMaterial: rho={rho}, Cp={Cp}, k={k} (Ti-6Al-4V)")
print(f"Beam:     r={r*1e6:.0f} um,  d={d*1e6:.0f} um  (delta={r/d:.1f})")
print()

# Show that P only scales Tc, never changes Pe
print("Key result: P scales Tc linearly — SAME Pe for all power levels")
print()
print(f"  {'P (W)':>8}  {'Tc (K)':>10}  {'Pe':>8}  "
      f"{'Same Pe?':>10}  Interpretation")
print("  " + "-" * 65)
Pe_ref = None
for P in [50, 100, 200, 300, 400]:
    Pe, delta, Tc = compute_groups(P, eta, 1.0, r, d, rho, Cp, k, T0)
    if Pe_ref is None:
        Pe_ref = Pe
    print(f"  {P:>8}  {Tc:>10.0f}  {Pe:>8.2f}  "
          f"{'YES' if abs(Pe-Pe_ref)<1e-6 else 'NO':>10}  "
          f"Same shape, Tc x{P/50:.0f}")

print()
print("  -> Changing P at fixed V is equivalent to rescaling T by P.")
print("     You only need ONE simulation per Pe value.")

# -----------------------------------------------------------------------
# Parametric map: (V, r, d) → (Pe, delta)
# -----------------------------------------------------------------------
print()
print("=" * 70)
print("PARAMETRIC MAP: (V, d) → (Pe, delta)  for r=50um, Ti-6Al-4V")
print("=" * 70)
print()

V_vals = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
d_vals = [25e-6, 50e-6, 100e-6, 200e-6, 250e-6]

print(f"  {'V (m/s)':>9}", end="")
for d_v in d_vals:
    print(f"  d={d_v*1e6:.0f}um", end="")
print()
print(f"  {'':>9}", end="")
for d_v in d_vals:
    print(f"  (δ={r/d_v:.2f})", end="")
print()
print("  " + "-" * 75)

for V_v in V_vals:
    Pe_v = rho * Cp * V_v * r / k
    print(f"  {V_v:>9.1f}", end="")
    for d_v in d_vals:
        Pe2, delta2, _ = compute_groups(200, eta, V_v, r, d_v, rho, Cp, k)
        print(f"  Pe={Pe2:5.1f}", end="")
    print(f"   (Pe={Pe_v:.1f})")

# -----------------------------------------------------------------------
# Domain check: when is semi-infinite assumption valid?
# -----------------------------------------------------------------------
print()
print("=" * 70)
print("DOMAIN CHECK — when is the domain large enough?")
print("Rule: domain > 10r (lateral), 10d (depth), V*t_total/r > 5 (time)")
print("=" * 70)
print()

cases = [
    ("Micro component",   0.5e-3, 0.5e-3, 0.1e-3, 50e-6),
    ("Single track",      5e-3,   1e-3,   0.3e-3, 50e-6),
    ("Layer scan",        20e-3,  5e-3,   0.5e-3, 50e-6),
    ("Part build",       100e-3, 50e-3,   2e-3,   50e-6),
]

print(f"  {'Case':>20}  {'Lx/r':>7}  {'Ly/r':>7}  {'Lz/d':>7}  "
      f"{'ts_max@1m/s':>12}  Semi-inf?")
print("  " + "-" * 75)
for name, Lx, Ly, Lz, r_v in cases:
    xs = Lx / r_v
    ys = Ly / r_v
    zs = Lz / r_v   # d = r here
    ts = 1.0 * 1e-3 / r_v  # t_total = 1ms, V=1m/s
    ok = xs > 10 and ys > 10 and zs > 10
    print(f"  {name:>20}  {xs:>7.0f}  {ys:>7.0f}  {zs:>7.0f}  "
          f"{ts:>12.0f}  {'YES' if ok else 'NO (BCs matter)'}")

# -----------------------------------------------------------------------
# Reconstruction example: one Pe solution → multiple (P, V) results
# -----------------------------------------------------------------------
print()
print("=" * 70)
print("RECONSTRUCTION: one Pe=17 solution → many dimensional cases")
print("  Ts_peak = 0.22 (hypothetical dimensionless peak temperature)")
print("=" * 70)
print()

Ts_peak = 0.22   # from a single Pe=17, delta=1 simulation

print(f"  {'P (W)':>8}  {'V (m/s)':>9}  {'Pe':>6}  "
      f"{'Tc (K)':>8}  {'T_peak (K)':>12}  Melt? (Tm=1878K)")
print("  " + "-" * 70)

for P_v in [100, 200, 300, 400]:
    for V_v in [0.5, 1.0, 2.0]:
        Pe_v, _, Tc_v = compute_groups(P_v, eta, V_v, r, r, rho, Cp, k)
        if abs(Pe_v - 17.7) < 5:   # only show Pe ~ 17 cases
            Tpeak = reconstruct_T(Ts_peak, Tc_v, T0)
            melt = "YES" if Tpeak > 1878 else "no"
            print(f"  {P_v:>8}  {V_v:>9.1f}  {Pe_v:>6.1f}  "
                  f"{Tc_v:>8.0f}  {Tpeak:>12.0f}  {melt}")

print()
print("  All rows have Pe≈17 → same dimensionless solution Ts(xs,ys,zs,ts)")
print("  T_peak varies only because Tc = P*eta/(k*r) is proportional to P.")
print()
print("  CONCLUSION: Run simulations on the (Pe, delta) grid.")
print("  Store Ts(xs,ys,zs,ts). Reconstruct T for any (P, V, r, material)")
print("  by computing Tc and Pe, then T = T0 + Tc * Ts.")
