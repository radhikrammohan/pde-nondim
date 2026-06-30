"""
Example 6: Multiple scaling strategies
(Langtangen & Pedersen §3.4.1 — convection-diffusion equation)

  ∂u/∂t + V ∂u/∂x = α ∂²u/∂x²

Two natural time scales (§3.4.1):
  1. Convective:  tc = L/V     →  ∂ūs/∂ts + ∂ūs/∂xs = Pe⁻¹ ∂²ūs/∂xs²
  2. Diffusive:   tc = L²/α   →  ∂ūs/∂ts + Pe ∂ūs/∂xs = ∂²ūs/∂xs²

"For large Pe, the convective time scale is most appropriate …
 Only when the diffusion term is much larger than convection (small Pe)
 is tc = L²/α the right time scale."
                                          — Langtangen & Pedersen §3.4.1
"""
import sympy as sp
from pde_nondim import MultipleScalings, suggest_scales

x, t = sp.symbols("x t", positive=True)
V, alpha, L, C0 = sp.symbols("V alpha L C0", positive=True)
tc = sp.Symbol("tc", positive=True)   # unknown to solve for

u = sp.Function("u")(x, t)

pde = sp.Eq(
    sp.diff(u, t) + V * sp.diff(u, x),
    alpha * sp.diff(u, x, 2),
)

print("Dimensional PDE:")
print(" ", pde)
print()

# ------------------------------------------------------------------
# Scale suggestion: what does "balance of terms" give for tc?
# ------------------------------------------------------------------
print("Scale suggestions via balance of terms (unknown: tc):")
candidates = suggest_scales(
    pde=pde,
    known_scales={u: C0, x: L, t: tc},   # tc is a free symbol — we solve for it
    unknown_scales=[tc],
)
for desc, assignment in candidates:
    print(" ", desc)
print()

# ------------------------------------------------------------------
# Compare both scalings side-by-side (book eqs. 3.69 and 3.70)
# ------------------------------------------------------------------
ms = MultipleScalings(
    pde=pde,
    scale_options=[
        {u: C0, x: L, t: L / V},           # convective tc
        {u: C0, x: L, t: L**2 / alpha},    # diffusive tc
    ],
    labels=[
        "Convective  tc = L/V   (appropriate when Pe >> 1)",
        "Diffusive   tc = L²/α  (appropriate when Pe << 1)",
    ],
    # For the diffusive case, normalise by the diffusion term (last term,
    # index -1) so we get ∂ūs/∂ts + Pe·∂ūs/∂xs = ∂²ūs/∂xs²  (book eq. 3.70)
    reference_term=["first", -1],
)
ms.print_all()
