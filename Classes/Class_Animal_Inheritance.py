# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:04:13 2019

@author: tulay.caglayan
"""

# parent
class Animal:
    
    def __init__(self):
        print("animal is created ")
        
    def toString (self):
        print("animal")
     
    def walk (self):
        print("animal walk")
        
#child 
class Monkey(Animal):
    
    def __init__(self):
        super().__init__() # use inot of parent class
        print("monkey is created ")
        
    def toString (self):
        print("monkey")
    
    def climb (self):
        print("monkey can climb")    
        
#child 
class Bird(Animal):
    
    def __init__(self):
        super().__init__() # use inot of parent class
        print("bird is created ")
        
    def toString (self):
        print("bird")
    
    def fly (self):
        print("bird can fly")    
        
m1 = Monkey()

m1.walk()
m1.toString()
m1.climb()

b1 = Bird()

b1.walk()
b1.toString()
b1.fly()