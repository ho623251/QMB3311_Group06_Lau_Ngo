"""
##################################################
#
# QMB 3311: Python for Business Analytics
#
# Name: Ashley Lau and Vu Minh Thu Ngo
#
# Date: January 14
#
##################################################
#
# Script for Assignment 2:
# Function Definitions
#
##################################################
"""


##################################################
# Import Required Modules
##################################################

# import name_of_module


##################################################
# Function Definitions
##################################################

# Exercise 1
def present_value(cash_flow: float, interest_rate: float, num_yrs: float) -> float:
    
    
    answer_present = cash_flow / (1+ interest_rate) ** num_yrs 

    return answer_present

# Testing examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 1.")

print("#" + 50*"-")
print("Exercise 1, Example 1:")
print("Evaluating present_value(1000, 0.05, 1)")
print("Expected: " + str(952.38))
print("Got: " + str(present_value(1000, 0.05, 1)))

""
print("#" + 50*"-")
print("Exercise 1, Example 2:")
print("Evaluating present_value(5000, 0.05, 2)")
print("Expected: " + str(4535.15))
print("Got: " + str(present_value(5000 , 0.05, 2)))


print("#" + 50*"-")
print("Exercise 1, Example 3:")
print("Evaluating present_value(10000, 0.10, 3)")
print("Expected: " + str(7513.15))
print("Got: " + str(present_value(10000, 0.10, 3)))


# Exercise 2
def future_value(cash_flow: float, interest_rate: float, num_yrs: float) -> float:
    
    answer_future = cash_flow * (1+ interest_rate) ** num_yrs 

    return answer_future


# Testing examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 2.")

print("#" + 50*"-")
print("Exercise 2, Example 1:")
print("Evaluating future_value(1000, 0.05, 1)")
print("Expected: " + str(1050))
print("Got: " + str(future_value(1000, 0.05, 1)))

print("#" + 50*"-")
print("Exercise 2, Example 2:")
print("Evaluating future_value(5000,0.05,2)")
print("Expected: " + str(5512.5))
print("Got: " + str(future_value(5000,0.05,2)))

print("#" + 50*"-")
print("Exercise 2, Example 3:")
print("Evaluating future_value(10000, 0.1, 3)")
print("Expected: " + str(13310))
print("Got: " + str(future_value(10000, 0.1, 3)))


#Exercise 3
def total_revenue(units_sold: int, price: float):
    total = units_sold * price 
    
    return total

# Testing examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 3.")

print("#" + 50*"-")
print("Exercise 3, Example 1:")
print("Evaluating total(3,5)")
print("Expected: " + str(15))
print("Got: " + str(total_revenue(3,5)))

print("#" + 50*"-")
print("Exercise 3, Example 2:")
print("Evaluating total(4,10.5)")
print("Expected: " + str(42))
print("Got: " + str(total_revenue(4,10.5)))

print("#" + 50*"-")
print("Exercise 3, Example 3:")
print("Evaluating total(10,11)")
print("Expected: " + str(110))
print("Got: " + str(total_revenue(10,11)))

# Exercise 4 
def total_cost(quantity_produced, fixed_cost: float, k):
    
    cal = k * (quantity_produced ** 2) + fixed_cost
    
    return cal

# Testing examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 4.")

print("#" + 50*"-")
print("Exercise 4, Example 1:")
print("Evaluating total(3,5,2)")
print("Expected: " + str(23))
print("Got: " + str(total_cost(3,5,2)))

print("#" + 50*"-")
print("Exercise 4, Example 2:")
print("Evaluating total(10,5,9.5)")
print("Expected: " + str(955))
print("Got: " + str(total_cost(10,5,9.5)))

print("#" + 50*"-")
print("Exercise 4, Example 3:")
print("Evaluating total(4,5,10)")
print("Expected: " + str(165))
print("Got: " + str(total_cost(4,5,10)))

# Exercise 5

def CESutility(x, y, r):
    
    # Compute the CES utility function for r ≠ 0
    cal = (x**r + y**r)**(1 / r)
    return cal

# Testing examples
print("#" + 50*"-")
print("Testing my Examples for Exercise 5.")

print("#" + 50*"-")
print("Exercise 5, Example 1:")
print("Evaluating total(3,5,2)")
print("Expected: " + str(23))
print("Got: " + str(total_cost(3,5,2)))

print("#" + 50*"-")
print("Exercise 4, Example 2:")
print("Evaluating total(5,9,5)")
print("Expected: " + str(134))
print("Got: " + str(total_cost(5,9,5)))

print("#" + 50*"-")
print("Exercise 4, Example 3:")
print("Evaluating total(3,4,5")
print("Expected: " + str(49))
print("Got: " + str(total_cost(3,4,5)))
