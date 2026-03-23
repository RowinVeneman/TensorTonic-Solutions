import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if len(x) != len(p):
        raise ValueError("x and p are not the same size")
    if abs(1-sum(p)) > 0.000001:
        raise ValueError("probabilities do not sum to 1")

    return sum([x_i*p_i for (x_i, p_i) in zip(x,p)])
        
