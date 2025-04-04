##################################################
#
# QMB 3311: Python for Business Analytics
#
# Name: Ashley Lau and Vu Minh Thu Ngo
#
# Date: 14th March 2025
# 
##################################################
#
# Sample Script for Assignment 5: 
# Function Definitions
#
##################################################


##################################################
# Import Required Modules
##################################################

import math
import numpy as np
from typing import List

##################################################
# Function Definitions
##################################################

# Only function definitions here - no other calculations. 

#--------------------------------------------------
# Question 1
# Functions from Previous Assignments
#--------------------------------------------------
# Start with the function from previous assignments.

# Assignment 3, question c
def logit(x: float, beta0: float, beta1: float) -> float:
    """
    Computes the logit link function l(x; β0, β1) = exp(β0 + x * β1) / (1 + exp(β0 + x * β1)).
    This is commonly used in logistic regression for binary classification problems.
    
    The inputs are:
    - x: The feature value (e.g., input variable)
    - β0: The intercept term
    - β1: The coefficient for the feature x
    
    The output is the probability that y = 1 given the feature x.
    
    >>> logit(1, 0, 1)
    0.7310585786300049
    >>> logit(2, 0, 1)
    0.8807970779778823
    >>> logit(0, 0, 1)
    0.5
    """
    z = beta0 + x * beta1
    return 1 / (1 + math.exp(-z))

# Assignment 3, question d
def logit_like(y: int, x: float, beta0: float, beta1: float) -> float:
    """
    Calculate the log-likelihood for logistic regression observation (y, x).
    
    Arguments:
    y -- binary outcome (either 0 or 1)
    x -- independent variable (float)
    beta0 -- intercept (float)
    beta1 -- coefficient for x (float)
    
    Returns:
    log-likelihood (float)
    
    Examples:
    >>> logit_like(1, 1, 0, 2) 
    -0.12692801104297263
    >>> logit_like(0, 1, 0, 2) 
    -2.1269280110429714
    >>> logit_like(1, 0, 0, 1) 
    -0.6931471805599453
    """
    prob = logit(x, beta0, beta1)
    if y == 1:
        return math.log(prob)
    elif y == 0:
        return math.log(1 - prob)
    else:
        raise ValueError("y must be either 0 or 1.")
        
# Assignment 4, question b       
def logit_like_sum(y: np.ndarray, x: np.ndarray, beta0: float, beta1: float) -> float:
    """
    Computes the total log-likelihood sum for a logistic regression model.

    This function aggregates the log-likelihood contributions across all observations 
    using the logit_like() function.

    Arguments:
    y -- binary outcomes (numpy array or list of 0's and 1's)
    x -- predictor variables (numpy array or list of float values)
    beta0 -- intercept term (float)
    beta1 -- coefficient for the predictor variable (float)
    
    Returns:
    float
        The sum of the log-likelihood across all observations.

    Examples:
    y1 = np.array([1, 0, 1, 1])
    x1 = np.array([2.0, 1.5, 3.0, 2.5])
    beta0_1 = 0.5
    beta1_1 = -0.3
    logit_like_sum(y1, x1, beta0_1, beta1_1) -> -3.2018109803656536

    y2 = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    x2 = np.array([2.1, 1.2, 2.8, 3.0, 1.5, 2.0, 2.6, 1.8, 3.2, 2.3])
    beta0_2 = -0.2
    beta1_2 = 0.4
    logit_like_sum(y2, x2, beta0_2, beta1_2) -> -7.699995345786197

    y3 = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    x3 = np.array([1.5, 2.2, 1.0, 2.8, 3.5, 2.0, 3.1, 2.4, 1.8, 3.3])
    beta0_3 = 1.0
    beta1_3 = -1.2
    logit_like_sum(y3, x3, beta0_3, beta1_3) -> -10.946140604927246
    """
    
    if len(y) != len(x):
        raise ValueError("The lengths of y and x do not match.")
    log_likelihood_sum = 0.0
    for i in range(len(y)):
        log_likelihood_sum += logit_like(y[i], x[i], beta0, beta1)
    return log_likelihood_sum

#--------------------------------------------------
# Question 2
# New Functions
#--------------------------------------------------

# Assignment 5, question a   
def logit_di(x_i: float, k: int) -> float:
    """
    Computes the term d_i in the gradient vector for logistic regression.
    
    - If k = 0, returns 1.
    - If k = 1, returns x_i.
    - Otherwise, raises a ValueError.
    
    Arguments:
    x_i -- A single observation (float)
    k -- Integer indicator (0 or 1)
    
    Returns:
    float -- Either 1 or x_i, depending on k.
    
    Examples:
    >>> logit_di(2, 1) 
    2
    >>> logit_di(7, 0)
    1
    >>> logit_di(0.5, 1)
    0.5
    >>> logit_di(3, 2)  # This should raise an error
    Traceback (most recent call last):
        ...
    ValueError: k must be 0 or 1.
    """
    if k == 0:
        return 1
    elif k == 1:
        return x_i
    else:
        raise ValueError("k must be 0 or 1.")

# Assignment 5, question b
def logit_dLi_dbk(y_i: float, x_i: float, k: int, beta_0: float, beta_1: float) -> float:
    """
    Computes the partial derivative of log-likelihood for logistic regression.

    Arguments:
    y_i -- observed binary outcome (0 or 1)
    x_i -- independent variable
    beta_0 -- intercept
    beta_1 -- coefficient for x
    k -- index (0 for beta_0, 1 for beta_1)

    Returns:
    Gradient term dLi/dbk

    Examples:
    >>> logit_dLi_dbk(1, 0, 0, 0, 0)
    0.5
    >>> logit_dLi_dbk(0, math.log(15), 0, 16, 0)
    -1.0
    >>> logit_dLi_dbk(1, 2, math.log(11), math.log(3), 0)
    0.01
    >>> logit_dLi_dbk(0, 2, math.log(11), math.log(3), 1)
    -1.98
    >>> logit_dLi_dbk(18.5, 2, 1, 0, 3)
    Error: Observations in y must be either zero or one.
    """
    if y_i in [0, 1] :
        dLi_dbk = logit_di(x_i, k)*(y_i - logit(x_i, beta_0, beta_1))
    else:
            print('Error: Observations in y must be either zero or one.')
            dLi_dbk =  None
    return dLi_dbk


##################################################
# Test the examples in your docstrings
##################################################


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())


##################################################
# End
##################################################
