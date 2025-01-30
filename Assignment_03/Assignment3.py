# -*- coding: utf-8 -*-
"""
##################################################
#
# QMB 3311: Python for Business Analytics
#
# Name: Ashley Lau and Vu Minh Thu Ngo
#
# Date: January 21, 2025
# 
##################################################
#
# Sample Script for Assignment 3: 
# Function Definitions
#
##################################################
"""


##################################################
# Import Required Modules
##################################################

# import name_of_module
import math



##################################################
# Function Definitions
##################################################

# Only function definitions here - no other calculations. 

# Exercise 1

def CESutility_valid(x, y, r):
    if x < 0:
        print("Error: It must be positive number.")
        return None
    if y < 0:
        print("Error: It must be positive number.")
        return None
    if r <= 0:
        print("Error: It must be strickly positive number.")
        return None

    utility = (x**r + y**r)**(1 / r)
    return utility


# Excerise 2 

def CESutility_in_budget(x, y, r, px, py, w):
    if px < 0:
        print("Error: the price of x cannot be negative.")
        return None
    if py < 0:
        print("Error: the price of y cannot be negative.")
        return None
    if w < 0:
        print("Error: w (wealth) cannot be negative.")
        return None
    if r <= 0:
        print("Error: r must be strictly positive.")
        return None
    if px * x + py * y > w:
        print("Error: The chosen basket exceeds the budget.")
        return None
    
    return CESutility_valid(x, y, r)

#Excerise 3 

def logit(x, beta0, beta1):
    exponent = beta0 + beta1 * x
    return math.exp(exponent) / (1 + math.exp(exponent))

#Excerise 4 

def logit_like(y, x, beta0, beta1):
    prob = logit(x, beta0, beta1)
    if y == 1:
        return math.log(prob)
    elif y == 0:
        return math.log(1 - prob)
    else:
        return ValueError("y must be either 0 or 1.")



# Only function definitions above this point. 


##################################################
# Run the examples to test these functions
##################################################


# Excerise 1 examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 1.")

print("#" + 50*"-")
print("Exercise 1, Example 1:")
print("Evaluating CESutility_valid(1, -1, 5)")
print("Expected: 'Error: It must be positive number' " )
print("Got: " + str(CESutility_valid(1, -1, 5)))

print("#" + 50*"-")
print("Exercise 1, Example 2:")
print("Evaluating CESutility_valid(-1, 1, 5)")
print("Expected: 'Error: It must be positive number' " )
print("Got: " + str(CESutility_valid(-1, 1, 5)))

print("#" + 50*"-")
print("Exercise 1, Example 3:")
print("Evaluating CESutility_valid(1, 1, 5)")
print("Expected: 1.149 " )
print("Got: " + str(CESutility_valid(1, 1, 5)))
##################################################



# Excerise 2 examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 2.")

print("#" + 50*"-")
print("Exercise 2, Example 1:")
print("Evaluating CESutility_in_budget(1, 3, 4, 4, 5, 20)")
print("Expected: 3" )
print("Got: " + str(CESutility_in_budget(1, 3, 4, 4, 5, 20)))


print("#" + 50*"-")
print("Exercise 2, Example 2:")
print("Evaluating CESutility_in_budget(1, 3, 4, -4, 5, 20)")
print("Expected: It must be positive number" )
print("Got: " + str(CESutility_in_budget(-1, 3, 4, -4, 5, 20)))

print("#" + 50*"-")
print("Exercise 2, Example 3:")
print("Evaluating CESutility_in_budget(1, 3, 4, 4, 5, 5)")
print("Expected: The chosen basket exceeds the budget" )
print("Got: " + str(CESutility_in_budget(-1, 3, 4, 4, 5, 5)))

##################################################


# Excerise 3 examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 3.")

print("#" + 50*"-")
print("Exercise 3, Example 1:")
print("Evaluating logit(1, 1, 1)")
print("Expected: 0.88" )
print("Got: " + str(logit(1, 1, 1)))

print("#" + 50*"-")
print("Exercise 3, Example 2:")
print("Evaluating logit(3, 2, 1)")
print("Expected: 0.99" )
print("Got: " + str(logit(3, 2, 1)))

print("#" + 50*"-")
print("Exercise 3, Example 3:")
print("Evaluating logit(1, 1, 5)")
print("Expected: 0.99" )
print("Got: " + str(logit(1, 1, 5)))
##################################################



# Excerise 4 examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 4.")

print("#" + 50*"-")
print("Exercise 4, Example 1:")
print("Evaluating logit_like(0, 1, 1, 1)")
print("Expected: -2.13" )
print("Got: " + str(logit_like(0, 1, 1, 1)))

print("#" + 50*"-")
print("Exercise 4, Example 2:")
print("Evaluating logit_like(0, 3, 2, 1)")
print("Expected: -5.00" )
print("Got: " + str(logit_like(0, 3, 2, 1)))

print("#" + 50*"-")
print("Exercise 4, Example 3:")
print("Evaluating logit_like(1, 1, 1, 5))")
print("Expected: -0.00 " )
print("Got: " + str(logit_like(1, 1, 1, 5)))

##################################################
# End
##################################################