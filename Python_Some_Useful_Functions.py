# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:10:24 2019

@author: tulay.caglayan
"""

#%%************************ Factoriel ****************
def factoriel (inp):
    
    fact = 1
    for i in range (1, inp +1 ):
        fact = fact * i
        
    return fact


print ('factoriel(3) = ' , factoriel(3))


#%% Reverse word 

def wordReverse (wd):
   return wd[::-1]

print ('wordReverse(''Istanbul'') = ' , wordReverse('Istanbul'))

#%%************************ Convert Time ****************
# Example : input 128 , output 2:8
#

import math
def convertTime (num):
    hour =  math.floor(num / 60 )
    minute =  num % 60
    
    return str(hour ) +':' + str(minute)


print ('convertTime(128) = ' , convertTime(128))    

#%%************************ Capitialize words **************** 
# Example : input= 'kod yazmak cok zevkli'   , ouput= 'Kod Yazmak Cok Zevkli' 

def capitializeInits (str):
    strList = str.split(' ')
    
    for i  in range (0 , len(strList) ):
        strList[i] = strList[i][0].upper() +  strList[i][1:]
    
    return " ".join(strList)

print(capitializeInits('kod yazmak cok zevkli'))

#%%************************  Compare Words ****************
# Example : input= 'ankara'=='kaarna'   , ouput= true
# Example : input= 'ankara'=='xaga'   , ouput= false
    
def compareWords(str1, str2):
    
    for i in str1:
        if i not in str2:
            return False
   
    return True    

print (compareWords('ankara', 'kaarna'))

#%%************************ Find Frequency of Letters ****************
# example input = "kkwcccddeee"  output 2k1w3c2d3e

def findFrequency (wd):
 
    for  i in range(0, len(wd)):
        
        i = 0 
        outStr = ''
        
        while i < len(wd):
            
            c = wd[i]
            
            j = i+1
            compressed = [1,c]
            
            while j < len(wd):
                if wd[j] == c:
                    compressed[0] += 1 
                else:
                    break
                j += 1 
            
            outStr += ''.join(map(str, compressed))
            i =j     
                    
    return outStr    


print (findFrequency("kkwcccddeee"))

#%%************************ Find missing X value ****************
# find X
# input '10-X = 4' output = 4
# input '1X *11 = 121' output = 1
# input '1X0/3 = 50' output = 5

def findLostX(inp):
    
    str1 , str2 = inp.split('=')[0], inp.split('=')[1]
    
    for i in range(0,10):
        if eval( str1.replace('X', str(i))) == eval( str2):
            return str(i)
        
        
print('10-X = 4  --> X= ', findLostX('10-X = 4'))
print('1X *11 = 121  --> X=  ', findLostX('1X *11 = 121' ))
print('1X0/3 = 50 --> X=  ', findLostX('1X0/3 = 50' ))

#%%************************  Array Rotate **************** 
# input = 2,3,4,5   --> ilk elemanin ddegerini baslangic indexi say ve buradan baslayarak yaz ve tum sayalari tamamla 
# output = 4523
# input = 4,5,6,7,8,9,10,11,12,13
# output = 89101112134567

def arrayRotation(inp):
    outLst =[]
    
    for i in range ( inp[0], len(inp)):
        outLst.append(str(inp[i]))
        
    for i in range (0, inp[0]):
        outLst.append(str(inp[i]))
        
    return ''.join(outLst)    

print(arrayRotation([2,3,4,5]))
print(arrayRotation([4,5,6,7,8,9,10,11,12,13]))

#%%************************  Find Array Pairs ****************
# input =[5,6,6,5,3,3]   --> ters cifti olmayan cifleri bul
# output = 3,3 
# input =[7,8,8,7,9,1,1,9]  --> cifti olmayan yok 
# output = ok

def arrayPairs(arr):
    
    arrPairs =[]
    i  = 0
    while i < len(arr):
        arrPairs.append(  [ str(arr[i]) , str(arr[i+1]) ] )
        i +=2
    
    depo = []    
    for c in arrPairs:
        if c[::-1] not in arrPairs :
            for l in c:
                depo.append(l)
        elif c == c[::-1] and arrPairs.count(i) <2 :
            for l in c:
                depo.append(l)
            
    if depo == []:
        return 'ok'
    
    return ','.join(depo)
   
print(arrayPairs([5,6,4,5,6,5,3,3,2,2]))


#%%    

