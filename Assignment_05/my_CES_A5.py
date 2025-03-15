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

# Assignment 2, question e
def CESutility(x: float, y: float, r: float) -> float:
    """Return the CES utility given x, y, and r.
    
    >>> CESutility(2, 3, 0.5)
    9.898979486
    >>> CESutility(4, 4, 1)
    8.0
    >>> CESutility(1, 1, -1)
    0.5
    """
    if r == 0:
        return math.exp((math.log(x) + math.log(y)) / 2) 
    return (x**r + y**r)**(1/r)

# Assignment 3, question a
def CESutility_valid(x: float, y: float, r: float) -> float:
    """
    Returns the CES utility function value if inputs are valid. If any input is invalid, returns None.
    
    The inputs are:
    - x: Quantity of good x (should be non-negative)
    - y: Quantity of good y (should be non-negative)
    - r: Substitution parameter (should be strictly positive)

    If the conditions are met, the CES utility formula u(x; y; r) = (x^r + y^r)^(1/r) is applied.
    If any of the inputs are invalid (negative x, y, or r <= 0), it will return None and print an error message.

    >>> CESutility_valid(2, 3, 0.5)
    9.898979485566358
    >>> CESutility_valid(-1, 3, 0.5)
    Error: x cannot be negative.
    >>> CESutility_valid(2, 3, -1)
    Error: r must be positive.
    >>> CESutility_valid(2, 3, 0)
    Error: r must be positive.
    """
    if x < 0:
        print("Error: x cannot be negative.")
        return None
    if y < 0:
        print("Error: y cannot be negative.")
        return None
    if r <= 0:
        print("Error: r must be positive.")
        return None
    return CESutility(x, y, r)

# Assignment 3, question b
def CESutility_in_budget(x: float, y: float, r: float, p_x: float, p_y: float, w: float) -> float:
    """
    Returns the CES utility if the inputs are valid and the consumer's basket of goods is within budget.
    The function checks:
    - Prices (p_x and p_y) must be non-negative
    - r must be strictly positive
    - The total cost (p_x * x + p_y * y) must not exceed the consumer's wealth (w)

    If any of these conditions are violated, the function will return None and print the relevant error message.
    
    >>> CESutility_in_budget(2, 3, 0.5, 1, 2, 10)
    9.898979485566358
    >>> CESutility_in_budget(2, 3, 0.5, -1, 2, 10)
    Error: Prices must be non-negative and r must be positive.
    >>> CESutility_in_budget(2, 3, 0.5, 1, 2, 5)
    Error: The goods are not within budget.
    """
    if p_x < 0 or p_y < 0 or r <= 0:
        print("Error: Prices must be non-negative and r must be positive.")
        return None
    if w < p_x * x + p_y * y:
        print("Error: The goods are not within budget.")
        return None
    return CESutility_valid(x, y, r)

#--------------------------------------------------
# Question 2
# New Functions
#--------------------------------------------------

# Assignment 5, question c
def CESdemand_calc(r: float, p_x: float, p_y: float, w: float) -> list:
    """
    Returns the optimal values of x_star and y_star that maximize the CES utility function
    subject to the budget constraint: p_x * x + p_y * y <= w.
    
    The formulas used are:
    x_star = (p_x^(1/(r-1))) / (p_x^(r/(r-1)) + p_y^(r/(r-1))) * w
    y_star = (p_y^(1/(r-1))) / (p_x^(r/(r-1)) + p_y^(r/(r-1))) * w
    
    Args:
    r (float): Substitution parameter.
    p_x (float): Price of good x.
    p_y (float): Price of good y.
    w (float): Consumer's wealth.
    
    Returns:
    list: [x_star, y_star] that maximize CES utility while staying within the budget.
    
    >>> CESdemand_calc(2, 1, 2, 10)
    [2.0, 4.0]
    >>> CESdemand_calc(1.5, 3, 1, 20)
    [6.428571428571429, 0.7142857142857142]
    >>> CESdemand_calc(3, 5, 1, 15)
    [2.753701454334703, 1.2314927283264858]
    """
    denominator = p_x**(r/(r-1)) + p_y**(r/(r-1))
    
   
    x_star = (p_x**(1/(r-1))) / denominator * w
    y_star = (p_y**(1/(r-1))) / denominator * w
    
    
    return [x_star, y_star]

# Assignment 5, question d

import numpy as np
from typing import List

def max_CES_xy(
        x_min: float, x_max: float, 
        y_min: float, y_max: float, 
        step: float, 
        r: float, p_x: float, p_y: float, w: float) -> List[float]:
    """
Performs a grid search to find the optimal bundle of goods x and y that maximizes the Constant Elasticity of Substitution (CES) utility function.
    
    The search is conducted over a grid of candidate values for x and y, where x ranges from `x_min` to `x_max` and y ranges from `y_min` to `y_max`, with values separated by `step`. The candidate values are generated using `np.arange()`.

    This function does not include error handling, as it is managed within the `CESutility_in_budget()` function and by `np.arange()` for generating the candidate grid.

    The behavior of the utility function depends on the elasticity parameter `r`:
    - When 0 < r < 1, the CES utility function has a maximum, and the grid search returns the optimal bundle.
    - When r > 1, the CES utility function has a minimum, and the grid search will return a *corner solution*, where most of the wealth is spent on one good.

    Args:
        x_min (float): Minimum value for x in the grid search.
        x_max (float): Maximum value for x in the grid search.
        y_min (float): Minimum value for y in the grid search.
        y_max (float): Maximum value for y in the grid search.
        step (float): Step size for creating the grid of candidate values for x and y.
        r (float): The elasticity of substitution parameter.
        p_x (float): Price of good x.
        p_y (float): Price of good y.
        w (float): The total wealth or budget.

    Returns:
        List[float]: The optimal bundle [x_star, y_star] that maximizes the CES utility function.
    
    Examples:
    
    >>> max_CES_xy(0, 12/2, 0, 12/4, 0.1, 1/2, 2, 4, 12)
    [4.0, 1.0]
    
    >>> max_CES_xy(0, 10/2, 0, 10/4, 0.05, 0.4, 3, 5, 10)
    [2.0, 0.8]
   
    >>> max_CES_xy(0, 8/2, 0, 8/4, 0.01, 0.001, 2, 4, 8)
    [2.0, 1.0]

    """
    
    
    x_list = np.arange(x_min, x_max, step)
    y_list = np.arange(y_min, y_max, step)
    
    max_CES = float('-inf')
    i_max = None
    j_max = None
    
    for i in range(len(x_list)):
        for j in range(len(y_list)):
            x_i = x_list[i]
            y_j = y_list[j]
            if p_x*x_i + p_y*y_j <= w:
                CES_ij = CESutility_in_budget(x_i, y_j, r, p_x, p_y, w)
            else:
                CES_ij = None
            if not CES_ij == None and CES_ij > max_CES:
                max_CES = CES_ij
                i_max = i
                j_max = j
                
    if (i_max is not None and j_max is not None):
        return [x_list[i_max], y_list[j_max]]
    else:
        print("No value of utility was higher than the initial value.")
        print("Choose different values of the parameters for x and y.")
        return None




##################################################
# Test the examples in your docstrings
##################################################


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())


##################################################
# End
##################################################
