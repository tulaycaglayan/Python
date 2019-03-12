# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:08:59 2019

@author: tulay.caglayan
"""

class Calculator(object):
    
    # initialize   
    def __init__(self, a, b):
        " initilaze values "
        self.value1 = a
        self.value2 = b
    
    def add(self):
        " addition "
        return self.value1 + self.value2
        
    def multiply (self):
        " multiplication  "
        return self.value1 * self.value2
    
    def divide (self):
        " multiplication  "
        return self.value1 / self.value2
    

print("Choose add(1) , multiply(2), divide(3) ")
selection = input("select 1 or 2 or 3: ")

v1 = input("first value: ")
v2 = input("second value: ")

c1 = Calculator( int(v1), int(v2))

if selection == 1:
    print("Add: {} ".format(c1.add()))
elif selection == 2: 
    print("Multiply {} ".format(c1.multiply()))
else:    
    print("Divide {} ".format(c1.divide()))


