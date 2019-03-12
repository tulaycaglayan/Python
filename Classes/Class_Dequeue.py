# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:38:25 2019

@author: tulay.caglayan

double ended queue . you can add and pop from top and bottom
"""

class Dequeue(object):
    
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
        
    def addFront(self, item):
        """
        insert item into front queue
        """
        self.items.insert(0,item)
    
    def addRear(self, item):
        """
        insert item into back queue
        """
        self.items.append(item)
    
    def removeRear(self):
        """
        get last item in queue
        """
        return self.items.pop()
    
    def removeFront(self):
        """
        get first item in queue
        """
        return self.items.pop(0)
    
    def size(self):
        """
        size of queue
        """
        return len(self.items)
    
    
que = Dequeue()
print(que.isEmpty())
que.addFront("ankara")
que.addFront("izmir")
que.addFront("canakkale")
que.addFront("Istanbul")
print(que.items)
print(que.size())
print(que.removeFront())
print(que.items)
print(que.removeFront())
print(que.items)
print(que.removeFront())
print(que.items)
print(que.removeFront())
print(que.items)
print(que.size())



print(que.isEmpty())
que.addRear("ankara")
que.addRear("izmir")
que.addRear("canakkale")
que.addRear("Istanbul")
print(que.items)
print(que.size())
print(que.removeRear())
print(que.items)
print(que.removeRear())
print(que.items)
print(que.removeRear())
print(que.items)
print(que.removeRear())
print(que.items)
print(que.size())




print(que.isEmpty())
que.addRear("ankara")
que.addRear("izmir")
que.addRear("canakkale")
que.addRear("Istanbul")
print(que.items)
print(que.size())
print(que.removeFront())
print(que.items)
print(que.removeFront())
print(que.items)
print(que.removeFront())
print(que.items)
print(que.removeFront())
print(que.items)
print(que.size())