# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:38:25 2019

@author: tulay.caglayan
"""

class Queue2Stack(object):
    
     # initialize (constructor)
    def __init__(self):
        """
        initialize
        """
        self.stack1 =[]
        self.stack2 =[]
       
    def enqueue(self, item):
        """
        Stacke data eklemek ancak queue yaratmak icin
        """
        self.stack1.append(item)
    
    
    def dequeue(self):
        """
        stack1den data al stack2 ye ekle
        """
        if not self.stack2:
            while len(self.stack1) >0:
                self.stack2.append(self.stack1.pop())
                
        return self.stack2.pop()
    
que = Queue2Stack()

que.enqueue("ankara")
que.enqueue("izmir")
que.enqueue("canakkale")
que.enqueue("Istanbul")

print(que.stack1)
print(que.dequeue())
print(que.stack2)
print(que.dequeue())
print(que.stack2)
print(que.dequeue())