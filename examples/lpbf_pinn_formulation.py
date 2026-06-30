"""
Example 16: PINN-optimal non-dimensionalisation for LPBF

REQUIREMENTS for a well-conditioned PINN
-----------------------------------------
  1. All network inputs   in [-1, 1]  (or at least O(1))
  2. Network output Ts    in  [0, 1]  (avoids saturation)
  3. PDE residual terms   all O(1)    (balanced loss)
  4. Heat source fixed in space       (avoids chasing a moving Gaussian)
  5. Pe (and delta) as network inputs (one PINN for all process conditions)

TRANSFORMATION PIPELINE
------------------------
Step 1: Physical coordinates  ->  Rosenthal dimensionless  (removes P, r, rho, Cp, k)
Step 2: Move to laser frame    xi = xs - ts                 (fixes heat source at xi=0)
Step 3: Multiply PDE by Pe     (balances advection vs diffusion: both O(1) for any Pe)
Step 4: Map domain to [-1,1]   (standard PINN practice)

FINAL DIMENSIONLESS PDE  [PINN training form]
----------------------------------------------
Steady-state in the laser frame:

  ∂²Ts/∂ξ² + ∂²Ts/∂η² + δ²·∂²Ts/∂ζ²  +  Pe·∂Ts/∂ξ
  + (2δ/π^(3/2))·exp(-2ξ²-2η²-2ζ²) = 0

Network:
  inputs  : (ξ̃, η̃, ζ̃)  in [-1,1]³  and (Pe, δ)  as parameters
  output  : Ts(ξ̃, η̃, ζ̃; Pe, δ)  in [0,1]  (dimensionless temperature)

Two dimensionless parameters span the entire LPBF process space:
  Pe  = ρ·Cp·V·r / k       (Péclet — scan speed control)
  δ   = r / d              (beam aspect ratio — keyhole control)

Reconstruction:
  T(x,y,z,t) = T₀ + [P·η/(k·r)] · Ts((x-Vt)/r, y/r, z/d;  Pe, δ)
                     ~~~~~~~~~~~~~~~
                     scales linearly with P — query net, multiply by P/P_ref

COORDINATE MAPPING  (semi-infinite domain -> [-1,1])
------------------------------------------------------
Physically the melt pool extends:
  ξ  ∈ [-C_back, C_front]   C ~ 3-5/sqrt(Pe)  (thermal length behind laser)
  η  ∈ [-C_side, C_side]
  ζ  ∈ [-C_depth, 0]

Map to [-1,1] with:
  ξ̃ = 2*(ξ - ξ_min)/(ξ_max - ξ_min) - 1    etc.

For the far field (|ξ|, |η|, |ζ| >> 1):
  Ts → 0 exponentially (Dirichlet BC on box boundaries)

LOSS FUNCTION
-------------
  L = w_pde * L_pde  +  w_ic * L_bc  +  w_data * L_data

  L_pde  = mean[ (residual)² ]     over N_c collocation points
  L_bc   = mean[ (Ts - 0)² ]       on domain boundaries
  L_data = mean[ (Ts - Ts_meas)² ] on any available data points

  Recommended weights:  w_pde = 1,  w_bc = 10  (enforce BCs strongly)

PDE RESIDUAL (what you implement in your autograd framework):
  R = d2Ts_dxi2 + d2Ts_deta2 + delta^2*d2Ts_dzeta2
    + Pe*dTs_dxi
    + (2*delta/pi^1.5)*exp(-2*xi^2 - 2*eta^2 - 2*zeta^2)
"""

import numpy as np
import math

print("=" * 65)
print("PINN FORMULATION FOR LPBF")
print("=" * 65)

# -----------------------------------------------------------------------
# 1. Physical -> dimensionless transformation
# -----------------------------------------------------------------------
print("""
STEP 1: DIMENSIONAL → DIMENSIONLESS SCALING
─────────────────────────────────────────────
  T  = T₀ + Tc·Ts          Tc = P·η/(k·r)   [absorbs P]
  x  = r·xs                                   [absorbs r]
  y  = r·η
  z  = d·ζ                                    [absorbs d]
  t  = (r/V)·ts                               [absorbs V]

  Dimensionless groups: Pe = ρCpVr/k,  δ = r/d
""")

print("""
STEP 2: MOVE TO LASER FRAME
─────────────────────────────
  ξ = xs - ts = (x - V·t)/r       [laser is always at ξ=0]

  Heat source fixed: exp(-2ξ²-2η²-2ζ²)  ← no longer moving
""")

print("""
STEP 3: MULTIPLY BY Pe  (balance all terms)
────────────────────────────────────────────
  Before:  ∂Ts/∂ts = (1/Pe)∇²Ts + (1/Pe)·source
           Problem: 1/Pe << 1 for LPBF (Pe=17 → coefficient=0.06)

  After (steady-state, ×Pe):
    ∂²Ts/∂ξ² + ∂²Ts/∂η² + δ²·∂²Ts/∂ζ² + Pe·∂Ts/∂ξ + source = 0
    ↑diffusion         ↑depth         ↑advection  ↑O(1)
    O(1)               O(1)           O(Pe)        O(1)

  Note: Pe·∂Ts/∂ξ term can be O(Pe)~17 if ∂Ts/∂ξ is O(1).
  For Pe>>1, this is the dominant balance (advection ≈ source).
  The diffusion terms are a small correction — they control melt pool width.
""")

# -----------------------------------------------------------------------
# 2. Domain bounds for typical LPBF
# -----------------------------------------------------------------------
print("STEP 4: DOMAIN BOUNDS")
print("─" * 65)

print("""
  The thermal field decays as exp(-Pe·|ξ|/2) in the wake.
  Characteristic thermal length behind laser: L_th ~ 2/Pe.

  For Pe=17:  L_th ~ 0.12 r  (very short wake in dimensionless coords)
              So ξ ∈ [-5, 2],  η ∈ [-3, 3],  ζ ∈ [-3, 0]  is sufficient.
""")

print(f"  {'Pe':>6}  {'ξ_back':>8}  {'ξ_front':>9}  {'η_half':>8}  {'ζ_depth':>9}")
print("  " + "-" * 50)
for Pe_v in [1, 2, 5, 10, 17, 50]:
    # Rosenthal wake length ~ 1/Pe, spread ~ 1
    xi_back = max(5.0, 10.0 / Pe_v)   # thermal wake behind laser
    xi_front = 2.0                      # far ahead = cool, small domain needed
    eta_half = max(3.0, 3.0)
    zeta_depth = max(3.0, 3.0)
    print(f"  {Pe_v:>6}  {xi_back:>8.1f}  {xi_front:>9.1f}  "
          f"{eta_half:>8.1f}  {zeta_depth:>9.1f}")

# -----------------------------------------------------------------------
# 3. Network architecture recommendation
# -----------------------------------------------------------------------
print("""
NETWORK ARCHITECTURE
─────────────────────
  Inputs  (5 neurons):  ξ̃, η̃, ζ̃ ∈ [-1,1]  and  Pe, δ  (normalised)
  Output  (1 neuron) :  Ts ∈ [0, 1]          (dimensionless temperature)

  Normalise Pe and δ before feeding to network:
    Pe_norm  = (Pe - Pe_min) / (Pe_max - Pe_min)       -> [0,1]
    δ_norm   = (δ  - δ_min)  / (δ_max  - δ_min)        -> [0,1]

  Architecture options:
    - Standard MLP: 6 hidden layers × 64 neurons, tanh activation
    - Fourier feature embedding for ξ (captures sharp gradients near source)
    - Modified MLP with Pe conditioning (FiLM layers or hypernetwork)

  Activation: tanh preferred (smooth second derivatives for PDE residual)
""")

# -----------------------------------------------------------------------
# 4. Residual — write it explicitly for implementation
# -----------------------------------------------------------------------
print("PDE RESIDUAL (implement in PyTorch/JAX autograd)")
print("─" * 65)
print("""
  def pde_residual(xi, eta, zeta, Pe, delta, model):
      Ts = model(xi, eta, zeta, Pe, delta)   # shape [N]

      # First-order derivatives (via autograd)
      dTs_dxi   = grad(Ts, xi)
      dTs_deta  = grad(Ts, eta)
      dTs_dzeta = grad(Ts, zeta)

      # Second-order derivatives
      d2Ts_dxi2   = grad(dTs_dxi,   xi)
      d2Ts_deta2  = grad(dTs_deta,  eta)
      d2Ts_dzeta2 = grad(dTs_dzeta, zeta)

      # Gaussian source (fixed at origin in laser frame)
      src = (2*delta / math.pi**1.5) * exp(-2*xi**2 - 2*eta**2 - 2*zeta**2)

      # Residual (should be zero)
      R = (d2Ts_dxi2 + d2Ts_deta2
           + delta**2 * d2Ts_dzeta2
           + Pe * dTs_dxi
           + src)
      return R
""")

# -----------------------------------------------------------------------
# 5. What Pe and delta ranges to train on
# -----------------------------------------------------------------------
print("TRAINING PARAMETER SPACE")
print("─" * 65)

rho, Cp, k, r = 4430, 560, 7.0, 50e-6

print(f"\n  For Ti-6Al-4V (r={r*1e6:.0f}μm):")
print(f"  {'Process':>20}  {'V (m/s)':>9}  {'d/r':>6}  {'Pe':>8}  {'δ':>6}")
print("  " + "-" * 55)
cases = [
    ("Slow conduction",   0.1,  1.0),
    ("Std conduction",    0.5,  1.0),
    ("Fast conduction",   1.0,  1.0),
    ("Keyhole d=2r",      1.0,  2.0),
    ("Deep keyhole d=5r", 2.0,  5.0),
    ("Very fast",         3.0,  1.0),
]
for name, V, d_over_r in cases:
    Pe = rho*Cp*V*r/k
    delta = 1.0/d_over_r
    print(f"  {name:>20}  {V:>9.1f}  {d_over_r:>6.1f}  {Pe:>8.1f}  {delta:>6.2f}")

print(f"""
  Recommended training range:
    Pe    ∈ [1, 60]   (log-spaced: 1, 2, 5, 10, 20, 40, 60)
    delta ∈ [0.1, 2]  (log-spaced: 0.1, 0.2, 0.5, 1.0, 2.0)

  → 7 × 5 = 35 parameter combinations in training set
    Inference: any (Pe, δ) in range, any P (just multiply Ts by Tc)
""")

# -----------------------------------------------------------------------
# 6. Reconstruction from PINN output
# -----------------------------------------------------------------------
print("RECONSTRUCTION FROM PINN OUTPUT")
print("─" * 65)
print("""
  Given PINN(ξ̃, η̃, ζ̃, Pe, δ) = Ts  (dimensionless temperature rise)

  For a new process condition (P, V, r, d, material):

    1.  Pe   = rho*Cp*V*r / k
        delta = r / d
        Tc   = P*eta / (k*r)

    2.  Map query point to laser frame:
        ξ   = (x - V*t) / r
        η   = y / r
        ζ   = z / d

    3.  Map to [-1,1]:
        ξ̃ = 2*(ξ - ξ_min)/(ξ_max - ξ_min) - 1   etc.

    4.  Ts = PINN(ξ̃, η̃, ζ̃, Pe_norm, δ_norm)

    5.  T(x,y,z,t) = T0 + Tc * Ts
                   = T0 + [P*eta/(k*r)] * Ts

  Changing P only scales Tc — no new PINN evaluation needed.
  Changing V: recompute Pe, re-query PINN.
""")
