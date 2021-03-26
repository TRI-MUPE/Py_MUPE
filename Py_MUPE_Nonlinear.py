# General nonlinear instantiation of the Minimum Unbiased Percent Error technique (MUPE) for 
# multiplicative error models, which utilizes Iteratively Re-weighted Least Squares (IRLS) 
# with weights equal to the squared inverse predictions from the prior iteration. Utilizes 
# the MINPACK library implementation of the Levenberg-Marquardt algorithm.
# Usage Example:
#   mdict = mupe_nonlinear(func=my_func, y=df['y'], X=df['x'], start=(('a',10), ('b',1)))
#     - 'func' must be a function you have defined that specifies the model form
#     - 'y' is the response variable
#     - 'X' is the driver variable
#     - 'start' is a tuple of tuples providing the initial guess, or starting point for optimization. Parameter
#       labels must match those used in 'func'. Whenever possible, provide values of the correct sign and order
#       of magnitude. For log-linear model forms, use the LOLS or PING solution as the initial guess.
# Returns a dictionary containing an lmfit.model.ModelResult object and accompanying details.
#
import numpy as np
from lmfit import Model, Parameters
#
def mupe_nonlinear(func, y, X, start):
    model = Model(func)        # create lmfit model from input function
    parameters = Parameters()  # initialize starting guess
    for p,v in start:
        parameters.add(name=p, value=v)
    coeffs_prior = np.array(list(parameters.valuesdict().values()))  # initialize prior coefficients
    w = [1]*y.size                                                   # initialize weights
    for i in range(200):
        LM = model.fit(y, X=X, params=parameters, weights=w, max_nfev=10)  # Levenberg-Marquardt optimization
        w = 1/LM.best_fit                                 # reset weights
        coeffs = np.array(list(LM.best_values.values()))  # coefficients of current solution
        if np.allclose(coeffs_prior, coeffs): break       # stop if converged
        coeffs_prior = coeffs                             # reset prior coefficients
        parameters = Parameters();  j = 0                 # reset starting guess
        for p,v in start:
            parameters.add(name=p, value=coeffs[j]);  j = j + 1
    return {'model':LM, 'start':start, 'mupe_iters':i}
