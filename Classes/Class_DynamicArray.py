# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:10:39 2019

@author: tulay.caglayan     


dynamic  adv. disadv.

adv. :
    Fast Lookup. Istenilen arrayde ki degeri elde etmesi
    Variable size : resizeable 

disadv. :
    Slow worst-case append : eger yer yoksa kapasiteyio arttirmak yavas
    Costly inserts and deletes 
"""
import ctypes  # in order to create new array 

class DynamicArray(object):
    
    # initialize (constructor)
    def __init__(self):
        self.n = 0 # number of items
        self.capacity = 1 # capacity
        self.A = self.make_array(self.capacity)
        
    def __len__(self):
        """
        returns length of array
        """
        return self.n    
        
    def __getitem__(self,k):
        """
        get kth item
        """
        if not 0 <= k < self.n:
            return IndexError(" k is out of bounds!")
        
        return self.A[k]
    
    def append(self , item):
        """
        add item to array 
        """
        if self.n == self.capacity: # capacity is full
            self._resize(2*self.capacity)
            
        self.A[self.n] = item # add item
        self.n +=1 # increase number of items as 1 
        
    def _resize(self, newCap) :
        """
        incraese capacity of array 
        """
        
        B = self.make_array(newCap)
        for k in range(self.n):
            B[k] = self.A[k]
            
        self.A = B
        self.capacity = newCap
    
    def make_array(self, capacity):
        """
        make array
        """
        return (capacity * ctypes.py_object)()
       

arr = DynamicArray ()
arr.append(1)
print (arr[0])
arr.append(2)
arr.append(3)
print (arr[0],arr[1],arr[2])









