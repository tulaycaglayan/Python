# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:38:25 2019

@author: tulay.caglayan
"""

class Queue(object):
    
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
    
    def enqueue(self, item):
        """
        insert item into queue
        """
        self.items.insert(0,item)
    
    
    def dequeue(self):
        """
        get last item in queue
        """
        return self.items.pop()
    
    def nextItem(self):
        """
        show last item in queue
        """
        return self.items[len(self.items)-1]
    
    def size(self):
        """
        size of queue
        """
        return len(self.items)
    
    
que = Queue()
print(que.isEmpty())
que.enqueue("ankara")
que.enqueue("izmir")
que.enqueue("canakkale")
que.enqueue("Istanbul")
print(que.size())
print(que.dequeue())
print(que.dequeue())
print(que.nextItem())
print(que.nextItem())
print(que.dequeue())
print(que.dequeue())
print(que.size())
