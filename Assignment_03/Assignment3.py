#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 23:27:52 2025
"""

##################################################
#
# QMB 3311: Python for Business Analytics
#
# Name: Ashley Lau and Vu Minh Thu Ngo
#
# Date: February 3rd, 2025
# 
##################################################
#
# Assignment 3: Function Definitions
#
##################################################

##################################################
# Import Required Modules
##################################################

import math

##################################################
# Part (a): CES Utility Function
##################################################

def CESutility_valid(x, y, r):
    if x < 0 or y < 0:
        print("Error: x and y must be positive numbers.")
        return None
    if r <= 0:
        print("Error: r must be strictly positive.")
        return None
    return (x**r + y**r)**(1 / r)

##################################################
# Part (b): CES Utility with Budget Constraint
##################################################

def CESutility_in_budget(x, y, r, px, py, w):
    if px < 0 or py < 0:
        print("Error: Prices cannot be negative.")
        return None
    if w < 0:
        print("Error: Wealth (w) cannot be negative.")
        return None
    if r <= 0:
        print("Error: r must be strictly positive.")
        return None
    if px * x + py * y > w:
        print("Error: The chosen basket exceeds the budget.")
        return None
    return CESutility_valid(x, y, r)

##################################################
# Part (c): Logit Function
##################################################

def logit(x, beta0, beta1):
    exponent = beta0 + beta1 * x
    return math.exp(exponent) / (1 + math.exp(exponent))

##################################################
# Part (d): Log-Likelihood of Logit Model
##################################################

def logit_like(y, x, beta0, beta1):
    prob = logit(x, beta0, beta1)
    if y == 1:
        return math.log(prob)
    elif y == 0:
        return math.log(1 - prob)
    else:
        raise ValueError("y must be either 0 or 1.")

##################################################
# Testing the Functions
##################################################

# Part (a) examples
print("#" + 50*"-")
print("Testing my Examples for Part (a).")

print("#" + 50*"-")
print("Part (a), Example 1:")
print("Evaluating CESutility_valid(2, -3, 1)")
print("Expected: 'Error: x and y must be positive numbers.'")  # y = -3 is negative, so the function should throw an error.
print("Got: " + str(CESutility_valid(2, -3, 1)))

print("#" + 50*"-")
print("Part (a), Example 2:")
print("Evaluating CESutility_valid(-2, 3, 1)")
print("Expected: 'Error: x and y must be positive numbers.'")  # x = -2 is negative, so the function should throw an error.
print("Got: " + str(CESutility_valid(-2, 3, 1)))

print("#" + 50*"-")
print("Part (a), Example 3:")
print("Evaluating CESutility_valid(2, 3, 1)")
print("Expected: 5.0")  # Using the CES utility formula: (2^1 + 3^1) = 5
print("Got: " + str(CESutility_valid(2, 3, 1)))

print("#" + 50*"-")
print("Part (a), Example 4:")
print("Evaluating CESutility_valid(2, 3, -1)")
print("Expected: 'Error: r must be strictly positive.'")  # r = -1 is invalid since r must be positive.
print("Got: " + str(CESutility_valid(2, 3, -1)))

##################################################



# Part (b) examples
print("#" + 50*"-")
print("Testing my Examples for Part (b).")

print("#" + 50*"-")
print("Part (b), Example 1:")
print("Evaluating CESutility_in_budget(2, 3, 1, 1, 1, 10)")
print("Expected: 5.0")  # Total cost: 1*2 + 1*3 = 5 (within budget of 10), so the utility is computed: (2^1 + 3^1) = 5
print("Got: " + str(CESutility_in_budget(2, 3, 1, 1, 1, 10)))

print("#" + 50*"-")
print("Part (b), Example 2:")
print("Evaluating CESutility_in_budget(2, 3, 1, -1, 1, 10)")
print("Expected: 'Error: Prices cannot be negative.'")  # Price px = -1 is negative, so the function should throw an error.
print("Got: " + str(CESutility_in_budget(2, 3, 1, -1, 1, 10)))

print("#" + 50*"-")
print("Part (b), Example 3:")
print("Evaluating CESutility_in_budget(2, 3, 1, 1, 1, 5)")
print("Expected: 5.0")  # Total cost: 1*2 + 1*3 = 5 (exactly equal to budget of 5), so the utility is computed: (2^1 + 3^1) = 5
print("Got: " + str(CESutility_in_budget(2, 3, 1, 1, 1, 5)))

print("#" + 50*"-")
print("Part (b), Example 4:")
print("Evaluating CESutility_in_budget(2, 3, 0.5, 1, 1, 10)")
print("Expected: 'Error: r must be strictly positive.'")  # r = 0.5 is valid, but should not be strictly positive.
print("Got: " + str(CESutility_in_budget(2, 3, 0.5, 1, 1, 10)))

##################################################



# Part (c) examples
print("#" + 50*"-")
print("Testing my Examples for Part (c).")

print("#" + 50*"-")
print("Part (c), Example 1:")
print("Evaluating logit(1, 0, 1)")
print("Expected: 0.731")  # Using the logit formula: logit(1) = e^(0 + 1*1) / (1 + e^(0 + 1*1)) ≈ 0.731
print("Got: " + str(logit(1, 0, 1)))

print("#" + 50*"-")
print("Part (c), Example 2:")
print("Evaluating logit(3, 2, 1)")
print("Expected: 0.9933")  # Using the logit formula: logit(3) = e^(2 + 1*3) / (1 + e^(2 + 1*3)) ≈ 0.9933
print("Got: " + str(logit(3, 2, 1)))

print("#" + 50*"-")
print("Part (c), Example 3:")
print("Evaluating logit(1, 1, 5)")
print("Expected: 0.9975")  # Using the logit formula: logit(1) = e^(1 + 5*1) / (1 + e^(1 + 5*1)) ≈ 0.9975
print("Got: " + str(logit(1, 1, 5)))

print("#" + 50*"-")
print("Part (c), Example 4:")
print("Evaluating logit(2, 0, 1)")
print("Expected: 0.8808")  # Using the logit formula: logit(2) = e^(0 + 1*2) / (1 + e^(0 + 1*2)) ≈ 0.8808
print("Got: " + str(logit(2, 0, 1)))

##################################################



# Part (d) examples
print("#" + 50*"-")
print("Testing my Examples for Part (d).")

print("#" + 50*"-")
print("Part (d), Example 1:")
print("Evaluating logit_like(1, 1, 0, 1)")
print("Expected: log(logit(1))")  # Since y = 1, it will compute the log of the probability for x = 1, β0 = 0, β1 = 1.
print("Expected output:", math.log(logit(1, 0, 1)))  # Should match the value of logit(1) ≈ 0.731
print("Got: " + str(logit_like(1, 1, 0, 1)))

print("#" + 50*"-")
print("Part (d), Example 2:")
print("Evaluating logit_like(0, 3, 2, 1)")
print("Expected: log(1 - logit(3, 2, 1))")  # Since y = 0, it will compute the log of (1 - probability for x = 3, β0 = 2, β1 = 1).
print("Expected output:", math.log(1 - logit(3, 2, 1)))  # Should match the value of (1 - logit(3)) ≈ 0.0067
print("Got: " + str(logit_like(0, 3, 2, 1)))

print("#" + 50*"-")
print("Part (d), Example 3:")
print("Evaluating logit_like(1, 1, 1, 5)")
print("Expected: log(logit(1))")  # Since y = 1, it will compute the log of the probability for x = 1, β0 = 1, β1 = 5.
print("Expected output:", math.log(logit(1, 1, 5)))  # Should match the value of logit(1) ≈ 0.9975
print("Got: " + str(logit_like(1, 1, 1, 5)))

print("#" + 50*"-")
print("Part (d), Example 4:")
print("Evaluating logit_like(0, 2, 0, 1)")
print("Expected: log(1 - logit(2, 0, 1))")  # Since y = 0, it will compute the log of (1 - probability for x = 2, β0 = 0, β1 = 1).
print("Expected output:", math.log(1 - logit(2, 0, 1)))  
print("Got: " + str(logit_like(0, 2, 0, 1)))

##################################################
# End
##################################################
