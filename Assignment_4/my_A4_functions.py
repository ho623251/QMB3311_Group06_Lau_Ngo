# -*- coding: utf-8 -*- 
"""
##################################################
#
# QMB 3311: Python for Business Analytics
#
# Name: Ashley Lau and Vu Minh Thu Ngo
#
# Date: 17th February 2025
# 
##################################################
#
# Sample Script for Assignment 4: 
# Function Definitions
#
##################################################
"""


##################################################
# Import Required Modules
##################################################

import numpy as np
import math

##################################################
# Function Definitions
##################################################

# Exercise 1(a): Define matrix_inverse function
def matrix_inverse(mat_in):
    """
    Compute the inverse of a 2x2 matrix.

    This function checks whether the given matrix is 2x2, calculates its determinant, 
    and returns the inverse if the determinant is nonzero. If the matrix is not 2x2 
    or if the determinant is zero, an error message is printed, and None is returned.

    The inverse of a 2x2 matrix:
    
        A = [[a, b], 
             [c, d]]
    
    is given by:
    
        A^-1 = (1 / det(A)) * [[d, -b], 
                               [-c, a]]
    
    where det(A) = (a*d - b*c).

    Parameters:
    ----------
    mat_in : numpy.ndarray
        A 2x2 NumPy array representing the input matrix.

    Returns:
    -------
    numpy.ndarray or None
        A 2x2 NumPy array representing the inverse of the input matrix if invertible.
        Returns None if the matrix is singular (determinant is zero) or not 2x2.

    Examples:
    --------
    >>> mat_in = np.array([[4, 7], [2, 6]])
    >>> matrix_inverse(mat_in)
    array([[ 0.6, -0.7],
           [-0.2,  0.4]])

    >>> mat_in = np.array([[3, 6], [1, 2]])
    >>> matrix_inverse(mat_in)
    Error: Determinant cannot be zero
    >>> None

    >>> mat_in = np.array([[5, 3], [2, 1]])
    >>> matrix_inverse(mat_in)
    array([[ 1., -3.],
           [-2.,  5.]])
    """
    # Check if the matrix is 2x2
    if mat_in.shape != (2, 2):
        print("Error: Matrix must be 2x2")
        return None
    
    # Calculate the determinant of the matrix
    det = mat_in[0, 0] * mat_in[1, 1] - mat_in[0, 1] * mat_in[1, 0]
    
    # If the determinant is zero, return None
    if det == 0:
        print("Error: Determinant cannot be zero")
        return None
    
    # Otherwise, calculate the inverse using the formula
    mat_out = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            mat_out[i, j] = ((-1)**(i + j) * mat_in[1 - i, 1 - j]) / det
    
    return mat_out


# Exercise 1(b): Compute the sum of the log-likelihood for logistic regression

def logit_like(y, x, beta_0, beta_1):
    """
    Computes the log-likelihood contribution for a single observation (y_i, x_i).

    The logistic regression model assumes that the probability of the event occurring 
    follows the logistic function:
    
        l(x; β0, β1) = exp(β0 + β1 * x) / (1 + exp(β0 + β1 * x))
    
    The log-likelihood for a single observation is computed as:
    
        L_i = y * log(l) + (1 - y) * log(1 - l)
    
    Parameters:
    -----------
    y : int
        Binary outcome (0 or 1).
    x : float
        Predictor variable.
    beta_0 : float
        Intercept term of the logistic regression model.
    beta_1 : float
        Slope coefficient for the predictor variable.
    
    Returns:
    --------
    float
        The log-likelihood contribution of the observation.

    Examples:
    ---------
    >>> logit_like(1, 2.0, 0.5, -0.3)
    -0.744396660073571

    >>> logit_like(0, 1.5, 0.5, -0.3)
    -0.7184596480132862
    """
    # Compute the probability using the logistic function
    logit_val = np.exp(beta_0 + beta_1 * x) / (1 + np.exp(beta_0 + beta_1 * x))
    
    # Compute log-likelihood for the observation
    return y * np.log(logit_val) + (1 - y) * np.log(1 - logit_val)

def logit_like_sum(y, x, beta_0, beta_1):
    """
    Computes the total log-likelihood sum for a logistic regression model.

    This function aggregates the log-likelihood contributions across all observations 
    using the logit_like() function.

    The log-likelihood function for the full dataset is:
    
        L(y, x; β0, β1) = Σ L_i(y_i, x_i; β0, β1)

    Parameters:
    -----------
    y : list or numpy.ndarray
        A list or array of binary outcome variables (0 or 1).
    x : list or numpy.ndarray
        A list or array of predictor variables.
    beta_0 : float
        The intercept term of the logistic regression model.
    beta_1 : float
        The coefficient for the predictor variable.
    
    Returns:
    --------
    float
        The sum of the log-likelihood across all observations.

    Examples:
    ---------
    Example 1:
    >>> y1 = np.array([1, 0, 1, 1])
    >>> x1 = np.array([2.0, 1.5, 3.0, 2.5])
    >>> beta_0_1 = 0.5
    >>> beta_1_1 = -0.3
    >>> logit_like_sum(y1, x1, beta_0_1, beta_1_1)
    -2.7018109803656536

    Example 2:
    >>> y2 = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    >>> x2 = np.array([2.1, 1.2, 2.8, 3.0, 1.5, 2.0, 2.6, 1.8, 3.2, 2.3])
    >>> beta_0_2 = -0.2
    >>> beta_1_2 = 0.4
    >>> logit_like_sum(y2, x2, beta_0_2, beta_1_2)
    -7.899995345786197

    Example 3:
    >>> y3 = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
    >>> x3 = np.array([1.5, 2.2, 1.0, 2.8, 3.5, 2.0, 3.1, 2.4, 1.8, 3.3])
    >>> beta_0_3 = 1.0
    >>> beta_1_3 = -1.2
    >>> logit_like_sum(y3, x3, beta_0_3, beta_1_3)
    -9.946140604927246
    """
    log_likelihood = beta_0  # Initialize log-likelihood sum with beta_0
    
    for i in range(len(y)):
        log_likelihood += logit_like(y[i], x[i], beta_0, beta_1)
    
    return log_likelihood


# Exercise 1(c): Define logit_like_grad function

def logit_like_grad(y: list, x: list, beta_0: float, beta_1: float) -> list:
    """
    Calculates the gradient vector of the likelihood function for the 
    bivariate logistic regression model with respect to the parameters 
    beta_0 and beta_1 for multiple observations.
    
    The gradient for each parameter (beta_0, beta_1) is calculated as:
    
    dL/d(beta_0) = sum(yi - l(x; beta_0, beta_1)) 
    dL/d(beta_1) = sum(xi * (yi - l(x; beta_0, beta_1)))
    
    Where:
    l(x; beta_0, beta_1) = 1 / (1 + exp(-(beta_0 + beta_1 * xi)))
    
    Args:
        y (list): List of binary outcomes (0 or 1).
        x (list): List of predictor values.
        beta_0 (float): Intercept parameter.
        beta_1 (float): Slope parameter.
        
    Returns:
        list: Gradient vector for beta_0 and beta_1.
    
    Examples:
    ---------
    >>> logit_like_grad([1, 1, 0, 0], [15.0, 5.0, 15.0, 5.0], 0.0, 0.0)
    [0.0, 0.0]
    
    >>> logit_like_grad([1, 1, 0, 0], [15.0, 5.0, 15.0, 5.0], math.log(3), 0.0)
    [-1.0, -10.0]
    
    >>> logit_like_grad([1, 1, 0, 0], [15.0, 5.0, 15.0, 5.0], math.log(7), 0.0)
    [-1.5, -15.0]
    
    >>> logit_like_grad([1, 0, 1], [1, 1, 1], 0.0, math.log(2))
    [0.0, 0.0]
    
    >>> logit_like_grad([1, 0, 1], [1, 1, 1], 0.0, math.log(5))
    [-0.5, -0.5]
    
    >>> logit_like_grad([1, 0, 1], [3, 3, 3], 0.0, math.log(2))
    [-2/3, -2.0]
    """
    
    # Calculate the logistic function
    def logistic_function(xi: float, beta_0: float, beta_1: float) -> float:
        return 1 / (1 + np.exp(-(beta_0 + beta_1 * xi)))

    # Initialize the gradients
    grad_beta_0 = 0.0
    grad_beta_1 = 0.0
    
    # Loop over all observations to compute the sum of the gradients
    for i in range(len(y)):
        logit_val = logistic_function(x[i], beta_0, beta_1)
        grad_beta_0 += (y[i] - logit_val)
        grad_beta_1 += (y[i] - logit_val) * x[i]
    
    # Return the gradient vector [grad_beta_0, grad_beta_1]
    return [grad_beta_0, grad_beta_1]


# Exercise 1(d): Define CESutility_multi function

def CESutility_multi(x, a, r):
    """
    Calculates the Constant Elasticity of Substitution utility for multiple goods,
    with a vector of quantities x and a vector of weights a. The parameter r represents
    the elasticity of substitution.
    
    The CES utility function is defined as:
    u(x, a; r) = {(sum_{i=1}^{n} a_i^(1-r) * x_i^r)}^(1/r)
    
    Where:
    x is a list of quantities of goods consumed,
    a is a list of weighting parameters for each good,
    r is the elasticity of substitution (parameter).
    
    Args:
        x (list): List of quantities of goods consumed. All elements must be non-negative.
        a (list): List of weighting parameters for each good. All elements must be non-negative.
        r (float): Elasticity of substitution parameter. Must not be 1.
    
    Returns:
        float: The calculated CES utility, or a string message indicating invalid input.
    
    Examples:
    ---------
    >>> CESutility_multi([1.0, 2.0], [1.0, 1.0], 0.5)
    5.82842712474619
    
    >>> CESutility_multi([1.0, -2.0], [0.5, 1.5], 0.5)
    'Error: x contains a negative value: -2.0'
    
    >>> CESutility_multi([1.0, 2.0], [1.0, 1.0], 1.0)
    'Error: r cannot be equal to 1, as the CES utility function is undefined for r = 1.'
    """
    
    # Step 1: Check if all elements in x and a are non-negative
    for i in x:
        if i < 0:
            return f"Error: x contains a negative value: {i}"
    
    for i in a:
        if i < 0:
            return f"Error: a contains a negative value: {i}"
    
    # Step 2: Check if the parameter r is equal to 1
    if r == 1:
        return "Error: r cannot be equal to 1, as the CES utility function is undefined for r = 1."
    
    # Step 3: Calculate the CES utility if all inputs are valid
    inside_sum = 0
    for i in range(len(x)):
        inside_sum += a[i] ** (1 - r) * x[i] ** r
    
    return inside_sum ** (1 / r)


##################################################
# Test the examples in your docstrings
##################################################

# Question 2: Test using the doctest module. 

# Exercise 1(a): Test log-likelihood sum for different datasets.

# Example 1: Matrix with non-zero determinant
mat_in_1 = np.array([[4, 7], [2, 6]])
print("Example 1: Inverse of mat_in_1:")
print(matrix_inverse(mat_in_1))  # Expected output: inverse matrix
print(np.linalg.inv(mat_in_1))   # Compare with numpy's inverse

# Example 2: Singular matrix (determinant is zero)
mat_in_2 = np.array([[3, 6], [1, 2]])
print("\nExample 2: Inverse of mat_in_2:")
print(matrix_inverse(mat_in_2))  # Expected output: Error message, None
try:
    print(np.linalg.inv(mat_in_2))  # Compare with numpy's inverse, should raise error
except np.linalg.LinAlgError as e:
    print(f"Error using numpy's inverse: {e}")

# Example 3: Another matrix with a non-zero determinant
mat_in_3 = np.array([[5, 3], [2, 1]])
print("\nExample 3: Inverse of mat_in_3:")
print(matrix_inverse(mat_in_3))  # Expected output: inverse matrix
print(np.linalg.inv(mat_in_3))   # Compare with numpy's inverse

# Exercise 1(b): Same as Exercise 1(a) - Identical examples for testing

# Example 1: Small dataset
y_test_1 = np.array([1, 0, 1, 1])
x_test_1 = np.array([2.0, 1.5, 3.0, 2.5])
beta_0_test_1 = 0.5
beta_1_test_1 = -0.3

print("Example 1: Log-Likelihood Sum =", logit_like_sum(y_test_1, x_test_1, beta_0_test_1, beta_1_test_1))


# Example 2: Larger dataset with different coefficients
y_test_2 = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
x_test_2 = np.array([2.1, 1.2, 2.8, 3.0, 1.5, 2.0, 2.6, 1.8, 3.2, 2.3])
beta_0_test_2 = -0.2
beta_1_test_2 = 0.4

print("Example 2: Log-Likelihood Sum =", logit_like_sum(y_test_2, x_test_2, beta_0_test_2, beta_1_test_2))


# Example 3: Testing with more extreme beta values
y_test_3 = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
x_test_3 = np.array([1.5, 2.2, 1.0, 2.8, 3.5, 2.0, 3.1, 2.4, 1.8, 3.3])
beta_0_test_3 = 1.0
beta_1_test_3 = -1.2

print("Example 3: Log-Likelihood Sum =", logit_like_sum(y_test_3, x_test_3, beta_0_test_3, beta_1_test_3))

# Exercise 1(c): Test log-likelihood gradient for different inputs.
print(logit_like_grad([1, 1, 0, 0], [15.0, 5.0, 15.0, 5.0], 0.0, 0.0))
print(logit_like_grad([1, 1, 0, 0], [15.0, 5.0, 15.0, 5.0], math.log(3), 0.0))
print(logit_like_grad([1, 1, 0, 0], [15.0, 5.0, 15.0, 5.0], math.log(7), 0.0))

# Exercise 1(d): Test CES utility function with various inputs.
print(CESutility_multi([1.0, 2.0], [1.0, 1.0], 0.5))  # Expected output: 5.82842712474619
print(CESutility_multi([1.0, -2.0], [0.5, 1.5], 0.5))  # Error: x contains a negative value: -2.0
print(CESutility_multi([1.0, 2.0], [1.0, 1.0], 1.0))  # Error: r cannot be equal to 1, as the CES utility function is undefined for r = 1.

##################################################
# End
##################################################
