
def f_der(a, x, b):
    return 2*a*x + b

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    xt = x0
    for step in range(steps):
        xt = xt - lr * f_der(a, xt, b)
    return xt