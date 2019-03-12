# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:06:19 2019

@author: tulay.caglayan
"""

class Node(object):
    
    def __init__(self , value):
        """
        Node yarat 
        """
        self.value = value
        self.nextnode = None
        
    def setNextNode (self, node):
        """
        Set next node
        """
        self.nextnode = node
    
    def getNextNode (self):
        """
        Get next node
        """
        return self.nextnode 
        
    
    def getNodeValue (self):
        """
        Get Node value
        """
        return self.value
        
    
ankara = Node("06")
canakkale = Node("17")
istanbul = Node("34")

ankara.setNextNode(canakkale)    
canakkale.setNextNode(istanbul)    

print(ankara.getNextNode().getNodeValue())
print(ankara.getNextNode().getNextNode().getNodeValue())