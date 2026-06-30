"""
Example 3: String-input convenience API

Same heat equation but entered as a string.
"""
import sympy as sp
from pde_nondim import NonDimensionalizer, parse_pde

eq, syms = parse_pde(
    "du/dt = alpha * d2u/dx2",
    functions=["u"],
    variables=["x", "t"],
    parameters=["alpha"],
)

print("Parsed PDE:", eq)
print("Symbols:", syms)
print()

x = syms["x"]
t = syms["t"]
u = syms["u"]

L, T_scale, Delta_T = sp.symbols("L T Delta_T", positive=True)
alpha = syms["alpha"]

nd = NonDimensionalizer(
    pde=eq,
    scales={u: Delta_T, x: L, t: T_scale},
    nd_suffix="s",
)

result = nd.run()
print(result)
