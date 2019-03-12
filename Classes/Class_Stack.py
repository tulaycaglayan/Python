# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:38:25 2019

@author: tulay.caglayan
"""

class Stack(object):
    
     # initialize (constructor)
    def __init__(self):
        """
        initialize
        """
        self.items =[]
       
    def isEmpty(self):
        """
        control if list is empty 
        """
        return self.items == [] #  boolean operation 
    
    def push(self, item):
        """
        insert item into stack
        """
        self.items.append(item)
    
    
    def pop(self):
        """
        get last item in stack
        """
        return self.items.pop()
    
    def top(self):
        """
        show last item in stack
        """
        return self.items[ len(self.items)-1]
    
    def size(self):
        """
        size of stack
        """
        return len(self.items)
    
    
stk = Stack()
print(stk.isEmpty())
stk.push("ankara")
stk.push("izmir")
stk.push("canakkale")
stk.push("Istanbul")
print(stk.size())
print(stk.pop())
print(stk.pop())
print(stk.top())
print(stk.top())