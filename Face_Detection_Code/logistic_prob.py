"""
 Princeton University, COS 429, Fall 2020
"""
from logistic_fit import logistic
def logistic_prob(X, params):
    """Given a logistic model and some new data, predicts probability that
       the class is 1.

    Args:
        X: datapoints (one per row, should include a column of ones
                       if the model is to have a constant)
        params: vector of parameters 

    Returns:
        z: predicted probabilities (0..1)
    """
    # Fill in here
    z = logistic(X @ params)
    return z
