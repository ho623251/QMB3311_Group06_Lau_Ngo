#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 22:28:01 2025

@author: ashleylau


##################################################
#
# QMB 3311: Python for Business Analytics
#
# Name:  Ashley Lau 
#
# Date: 03/31/2025
# 
##################################################
"""

import math
import doctest

# 1. Taylor Series Approximation for ln(z)
def ln_taylor(z, n):
    """
    Approximates ln(z) using the Taylor series expansion.

    >>> round(ln_taylor(1.5, 10), 5)
    0.40547
    >>> round(ln_taylor(2, 5), 5)
    0.69315
    """
    if z <= 0:
        raise ValueError("z must be greater than 0")
    approximation = 0
    for k in range(1, n + 1):
        approximation += ((-1) ** (k - 1)) * ((z - 1) ** k) / k
    return approximation

# 2. Function for exp(x) - z
def exp_x_diff(x, z):
    """
    Returns exp(x) - z.

    >>> round(exp_x_diff(math.log(2), 2), 5)
    0.0
    """
    return math.exp(x) - z

# 3. Bisection Method for ln(z)
def ln_z_bisect(z, a0, b0, num_iter):
    """
    Calculates ln(z) using the bisection method.

    >>> round(ln_z_bisect(2, 0, 2, 10), 5)
    0.69315
    """
    if exp_x_diff(a0, z) * exp_x_diff(b0, z) >= 0:
        raise ValueError("f(a0) and f(b0) must have opposite signs")
    for _ in range(num_iter):
        mid = (a0 + b0) / 2
        if exp_x_diff(mid, z) == 0:
            return mid
        elif exp_x_diff(a0, z) * exp_x_diff(mid, z) < 0:
            b0 = mid
        else:
            a0 = mid
    return mid

# 4. Derivative of exp(x) - z
def exp_x_diff_prime(x, z):
    """
    Returns the derivative of exp(x) - z with respect to x.

    >>> round(exp_x_diff_prime(0, 2), 5)
    1.0
    """
    return math.exp(x)

# 5. Newton's Method for ln(z)
def ln_z_newton(z, x0, tol, num_iter):
    """
    Calculates ln(z) using Newton's method.

    >>> round(ln_z_newton(2, 1, 1e-6, 10), 5)
    0.69315
    """
    for _ in range(num_iter):
        f_x = exp_x_diff(x0, z)
        f_prime_x = exp_x_diff_prime(x0, z)
        if abs(f_x) < tol:
            return x0
        x0 -= f_x / f_prime_x
    print("Warning: Maximum iterations reached.")
    return x0

# 6. Fixed-Point Iteration Function
def exp_x_fp_fn(x, z):
    """
    Returns the value of the fixed-point function g(x).

    >>> round(exp_x_fp_fn(1, 2), 5)
    0.5
    """
    return 0.5 * (z - math.exp(x) + 2 * x)

# 7. Fixed-Point Method for ln(z)
def ln_z_fixed_pt(z, x0, tol, num_iter):
    """
    Calculates ln(z) using the fixed-point method.

    >>> round(ln_z_fixed_pt(2, 1, 1e-6, 10), 5)
    0.69315
    """
    for _ in range(num_iter):
        x_next = exp_x_fp_fn(x0, z)
        if abs(x_next - x0) < tol:
            return x_next
        x0 = x_next
    print("Warning: Maximum iterations reached.")
    return x0

    doctest.testmod()
