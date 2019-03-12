# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:55:20 2019

@author: tulay.caglayan
"""

class Employee:
    
    def raisee(self):
        raise_rate = 0.1
        return 100 + 100 * raise_rate
    
class CompEng(Employee):
    
    def raisee(self):
        raise_rate = 0.2
        result =  100 + 100 * raise_rate    
        print("CompEng: ", result)

class EEE(Employee):
    
    def raisee(self):
        raise_rate = 0.3
        result =  100 + 100 * raise_rate    
        print("EEE: ", result)        
        

e1 = Employee()
ce = CompEng()
eee = EEE()

employee_list = [ce, eee]

for employee in employee_list:
    employee.raisee()