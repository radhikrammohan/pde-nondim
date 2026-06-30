"""
Example 14: LPBF — large Q₀ (high laser power / small beam) with phase change

When Q₀ is large, the characteristic temperature Tc = Q₀·r²/k becomes large.
Three new dimensionless groups emerge relative to the heat-only case:

  Θ_m = (T_melt - T₀)/Tc     -- melting threshold
  St   = L_f / (Cp·Tc)       -- Stefan number (latent heat / sensible heat)
  Θ_v  = (T_vap  - T₀)/Tc   -- vaporisation threshold  (optional)

When Θ_m < 1 (Q₀ large enough), the melt pool forms.
When St  >> 1, most energy goes into melting (temperature barely rises past Tm).
When St  << 1, the material "flies" past Tm — vaporisation / keyhole likely.

PDE (1-D, enthalpy formulation):
  ρ·[Cp + Lf·g(T)]·∂T/∂t = k·∂²T/∂x² + Q₀·exp(-2(x-Vt)²/r²)

where  g(T) = df_L/dT  is the derivative of liquid fraction (mushy-zone model):
  g(T) ≈ 1/ΔTm   within  [T_sol, T_liq],   ΔTm = T_liq - T_sol
  g(T) = 0        outside the mushy zone

Non-dimensionalisation
----------------------
  T = T₀ + Tc·Ts,    Tc = Q₀·r²/k
  x = r·xs,          t = (r/V)·ts
  g(T) → (Cp/Lf)·gs(Ts)      [gs ~ St within the mush]

Dimensionless PDE:
  (1 + gs(Ts))·∂Ts/∂ts = (1/Pe)·∂²Ts/∂xs² + (1/Pe)·exp(-2(xs-ts)²)

Three independent dimensionless parameters:
  Pe   = ρ·Cp·V·r / k            Péclet number
  Θ_m  = (T_m - T₀)·k / (Q₀·r²) Melting threshold   (< 1 → melt pool exists)
  St   = k·Lf / (Cp·Q₀·r²)      Stefan number        (= Lf/(Cp·Tc))

Note: Θ_m = St·(Cp/Lf)·(T_m - T₀) so they are not fully independent —
      Θ_m = St·Cp·(T_m-T₀)/Lf — but physically distinct.

Regimes:
  Q₀ small  Θ_m > 1     no melting,  heat-only problem
  Q₀ medium Θ_m < 1, St > 1   melt pool, latent heat limits T rise
  Q₀ large  Θ_m << 1, St << 1  melt pool + rapid T rise → keyhole risk
"""
import sympy as sp
import math
from pde_nondim import NonDimensionalizer

# -----------------------------------------------------------------------
# Symbols
# -----------------------------------------------------------------------
x, t = sp.symbols("x t", positive=True)
rho, Cp, k, Q0, r, V, T0 = sp.symbols("rho C_p k Q_0 r V T_0", positive=True)
Lf = sp.Symbol("L_f", positive=True)    # latent heat of fusion  [J/kg]
Tm = sp.Symbol("T_m", positive=True)    # melting temperature [K]

T = sp.Function("T")(x, t)
g = sp.Function("g")(T)    # dfL/dT  [1/K]

# Enthalpy-method heat equation with latent heat
pde = sp.Eq(
    rho * (Cp + Lf * g) * sp.diff(T, t),
    k * sp.diff(T, x, 2) + Q0 * sp.exp(-2 * (x - V * t)**2 / r**2),
)

print("=" * 65)
print("LPBF WITH PHASE CHANGE — large Q₀ regime")
print("=" * 65)
print("\nDimensional PDE (enthalpy formulation):")
print(" ", pde)
print("\n  g(T) = dfL/dT  (liquid fraction gradient, mushy zone model)")
print()

# -----------------------------------------------------------------------
# Non-dimensionalise
# g [1/K] × Lf [J/kg] has units [J/kgK] = same as Cp
# So natural scale for g is Cp/Lf  → Lf·g / Cp ~ O(1) in mushy zone
# -----------------------------------------------------------------------
Tc = Q0 * r**2 / k     # characteristic temperature

result = NonDimensionalizer(
    pde,
    scales={T: (Tc, T0), x: r, t: r / V},
    nonlinear_scales={g: Cp / Lf},
).run()

print(result)

# -----------------------------------------------------------------------
# Interpret the three groups
# -----------------------------------------------------------------------
print("\nAdditional groups from the large-Q₀ regime")
print("-" * 65)
print("These groups do NOT appear in the PDE directly — they enter")
print("through the BOUNDARY CONDITION (melting criterion) and the")
print("SHAPE of gs(Ts):\n")
print("  Pe  = ρ·Cp·V·r/k         (Péclet — from PDE coefficient)")
print("  Θ_m = (T_m - T₀)/Tc      (melting threshold)")
print("      = k·(T_m - T₀)/(Q₀·r²)")
print("        Θ_m < 1  =>  melt pool forms")
print("        Θ_m > 1  =>  no melting")
print()
print("  St  = L_f/(Cp·Tc)        (Stefan number)")
print("      = k·L_f/(Cp·Q₀·r²)")
print("        St >> 1  =>  latent heat dominates, T barely exceeds T_m")
print("        St << 1  =>  sensible heat dominates, T shoots well past T_m")
print()
print("  gs(Ts) in the dimensionless PDE has peak amplitude ~ St")
print("  within the mushy zone  Ts ∈ [Θ_sol, Θ_liq]")
print()
print("Dimensionless PDE:")
print("  (1 + gs(Ts))·∂Ts/∂ts = (1/Pe)·∂²Ts/∂xs²")
print("                        + (1/Pe)·exp(-2(xs-ts)²)")
print()
print("  -> For large Q₀:  St → 0,  Θ_m → 0")
print("     gs(Ts) collapses to zero amplitude (latent heat negligible)")
print("     Reduces back to the heat-only PDE but with T >> T_m")

# -----------------------------------------------------------------------
# Numerical analysis — Ti-6Al-4V across laser power levels
# -----------------------------------------------------------------------
print()
print("=" * 65)
print("NUMERICAL ANALYSIS — Ti-6Al-4V, three laser power regimes")
print("=" * 65)

rho_v  = 4430    # kg/m^3
Cp_v   = 560     # J/(kg.K)
k_v    = 7.0     # W/(m.K)
Lf_v   = 2.86e5  # J/kg   latent heat of fusion
T0_v   = 300     # K      ambient
Tm_v   = 1878    # K      solidus temperature Ti-6Al-4V
Tv_v   = 3560    # K      vaporisation temperature
r_v    = 50e-6   # m      beam radius
V_v    = 1.0     # m/s    scan speed

print(f"\n  r = {r_v*1e6:.0f} μm,  V = {V_v} m/s,  T₀ = {T0_v} K")
print(f"  Tm = {Tm_v} K,  Tv = {Tv_v} K,  Lf = {Lf_v:.2e} J/kg")
print()
print(f"  {'P (W)':>8}  {'Q₀ (W/m³)':>12}  {'Tc (K)':>8}  "
      f"{'Pe':>6}  {'Θ_m':>6}  {'St':>6}  {'Θ_v':>6}  Regime")
print("  " + "-" * 80)

for P_v, eta_v in [(50, 0.35), (200, 0.35), (500, 0.5), (1000, 0.6)]:
    Q0_v = 2 * P_v * eta_v / (math.pi**1.5 * r_v**3)
    Tc_v = Q0_v * r_v**2 / k_v
    Pe_v = rho_v * Cp_v * V_v * r_v / k_v
    Th_m = (Tm_v - T0_v) / Tc_v          # melting threshold
    St_v = k_v * Lf_v / (Cp_v * Q0_v * r_v**2)   # Stefan number = Lf/(Cp*Tc)
    Th_v = (Tv_v - T0_v) / Tc_v          # vaporisation threshold

    if Th_m > 1:
        regime = "no melting"
    elif Th_v > 1 and St_v > 0.5:
        regime = "melt pool (latent-heat limited)"
    elif Th_v < 1 and St_v < 0.1:
        regime = "KEYHOLE RISK (T >> Tv)"
    else:
        regime = "melt pool (transition)"

    print(f"  {P_v:>8}  {Q0_v:>12.2e}  {Tc_v:>8.0f}  "
          f"{Pe_v:>6.1f}  {Th_m:>6.3f}  {St_v:>6.3f}  {Th_v:>6.3f}  {regime}")

print()
print("  Θ_m < 1  => melt pool forms        (melting threshold crossed)")
print("  St  < 1  => sensible heat dominates (T rises well past T_m)")
print("  Θ_v < 1  => vaporisation possible   (T can reach T_vap)")

# -----------------------------------------------------------------------
# Regime map: Q₀ vs r
# -----------------------------------------------------------------------
print()
print("=" * 65)
print("REGIME MAP — melting (Θ_m=1) and keyhole (Θ_v=1) boundaries")
print("as a function of Q₀ for Ti-6Al-4V")
print("=" * 65)
print()

Q0_melt = k_v * (Tm_v - T0_v) / r_v**2   # Θ_m = 1
Q0_vapo = k_v * (Tv_v - T0_v) / r_v**2   # Θ_v = 1

print(f"  Q₀_melt = k·(Tm-T₀)/r²  = {Q0_melt:.2e} W/m³")
print(f"  Q₀_vap  = k·(Tv-T₀)/r²  = {Q0_vapo:.2e} W/m³")
print()
print(f"  For r = {r_v*1e6:.0f} μm:")
P_melt = Q0_melt * math.pi**1.5 * r_v**3 / (2 * 0.35)
P_vapo = Q0_vapo * math.pi**1.5 * r_v**3 / (2 * 0.5)
print(f"    Min power to melt    (eta=0.35): P > {P_melt:.0f} W")
print(f"    Min power to vaporise (eta=0.5): P > {P_vapo:.0f} W")
print()
print("  In general:")
print("    Q₀ < Q₀_melt   -> conduction only")
print("    Q₀_melt < Q₀ < Q₀_vap  -> stable melt pool")
print("    Q₀ > Q₀_vap    -> keyhole / vapour depression")
