import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    n,m = len(A), len(A[0])
    res = np.zeros((m,n))
    for i in range(n):
        for j in range(m):
            res[j,i] = A[i][j]
    return res