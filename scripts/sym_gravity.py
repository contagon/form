# ruff: noqa: F401, E741
from sympy import (
    Eq,
    fraction,
    symbols,
    Matrix,
    BlockMatrix,
    MatrixSymbol,
    Identity,
    ZeroMatrix,
    pretty_print as print,
    solve,
)

lam = symbols("lambda")

a, b, c = symbols("a b c")
m1, m2 = symbols("m1 m2")
g = symbols("g")

M = Matrix([[a, b], [b, c]])
W = Matrix([[0, 0], [0, 1]])
m = Matrix([[m1], [m2]])

x = 2 * (M + lam * W).inv()
_, poly = fraction(x[0, 0])

expr = (m.T * x * W * x.T * m)[0, 0]
expr = Eq(expr, g).simplify()
print(expr)

# solve for lam
sol = solve(expr, lam)
print(sol)
