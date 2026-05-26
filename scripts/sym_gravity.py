# ruff: noqa: F401, E741
from sympy import (
    Eq,
    Inverse,
    block_collapse,
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

# a, b, c = symbols("a b c")
# m1, m2 = symbols("m1 m2")
# g = symbols("g")

a = MatrixSymbol("A", 3, 3)
b = MatrixSymbol("B", 3, 3)
d = MatrixSymbol("D", 3, 3)

m1 = MatrixSymbol("m1", 3, 1)
m2 = MatrixSymbol("m2", 3, 1)
g = symbols("g")

Z = ZeroMatrix(3, 3)
I = Identity(3)

M = BlockMatrix([[a, b], [b.T, d]])
W = BlockMatrix([[Z, Z], [Z, I]])
m = BlockMatrix([[m1], [m2]])

x = 2 * (M + lam * W).inv()
x = block_collapse(x)

Q = MatrixSymbol("Q", 3, 3)
x = x.subs({Q: d - b.T * a.inv() * b})
print("---- x ----")
print(x)

print("\n\n\n--- x * W * x.T ---")
expr = block_collapse(x * W * x.T).simplify()
print(expr)
# expr = Eq(expr, g).simplify()
# print(expr)

# # solve for lam
# sol = solve(expr, lam)
# print(sol)


# # Define symbolic matrix symbols for the blocks
# A = MatrixSymbol("A", 2, 2)
# B = MatrixSymbol("B", 2, 2)
# C = MatrixSymbol("C", 2, 2)
# D = MatrixSymbol("D", 2, 2)

# # Construct the symbolic block matrix
# symbolic_block_matrix = BlockMatrix([[A, B], [C, D]])

# # Represent the inverse symbolically
# symbolic_inverse = Inverse(symbolic_block_matrix)

# print(block_collapse(symbolic_inverse))
