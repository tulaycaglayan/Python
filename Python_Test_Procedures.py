# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:10:24 2019

@author: tulay.caglayan
"""

#%% factoriel
def factoriel (inp):
    
    fact = 1
    for i in range (1, inp +1 ):
        fact = fact * i
        
    return fact


print ('factoriel(3) = ' , factoriel(3))


#%% Reverse word 

def wordReverese (wd):
    revWd =''
    for i in range(0, len(wd)):
        revWd =  wd[i] + revWd
        
    return revWd    

def wordReverese_udemy (str):
    # str [start:stop:step]
    return 


print ('wordReverese(''Istanbul'') = ' , wordReverese('Istanbul'))
print ('wordReverese_udemy(''Istanbul'') = ' , wordReverese_udemy('Istanbul'))

#%% Saat cevirme 
# Example : input 128 , output 2:8
#

import math
def convertTime (num):
    hour =  math.floor(num / 60 )
    minute =  num % 60
    
    return str(hour ) +':' + str(minute)


print ('convertTime(128) = ' , convertTime(128))    

#%% bas harf buyutme 
# Example : input= 'kod yazmak cok zevkli'   , ouput= 'Kod Yazmak Cok Zevkli' 

def capitializeInits (str):
    strList = str.split(' ')
    
    for i  in range (0 , len(strList) ):
        strList[i] = strList[i][0].upper() +  strList[i][1:]
    
    return " ".join(strList)

print(capitializeInits('kod yazmak cok zevkli'))

#%% kelime karistirma 
# Example : input= 'ankara'=='kaarna'   , ouput= true
# Example : input= 'ankara'=='xaga'   , ouput= false
    
def compareWords(str1, str2):
    
    for i in range(0 , len(str1)):
        if str1[i] not in str2:
            return False
   
    return True    

print (compareWords('ankara', 'kaarna'))
print (compareWords('ankara', 'xaga'))

    
def compareWords_udemy(str1, str2):
    
    for i in str1:
        if i not in str2:
            return False
   
    return True    

print (compareWords_udemy('ankara', 'kaarna'))
print (compareWords_udemy('ankara', 'xaga'))

#%% siklik bulma 
# example input = "kkwcccddeee"  output 2k1w3c2d3e

def siklikBulma (wd):
    
    letter = ''
    outStr = ''
    cnt = 0
    
    for  i in range(0, len(wd)):
        
        if letter == wd[i]:
           cnt = cnt +1 
           
        if letter != wd[i]:
           if letter != '':
              outStr +=  str(cnt) + letter
           letter = wd[i]  
           cnt = 1
           
    outStr = outStr + str(cnt) + letter       
    return outStr    


print (siklikBulma("kkwcccddeee"))


def siklikBulma_udemy (wd):
 
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


print (siklikBulma_udemy("kkwcccddeee"))

#%% kayip basamak
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



def findLostX_Udemy(inp):
    
    for i in range(10):
        c = inp.replace('X', str(i))
        x = inp.index('=')
        if eval( c[:x]) == eval( c[x+1:]):
            return str(i)
        
print('10-X = 4  --> X= ', findLostX_Udemy('10-X = 4'))
print('1X *11 = 121  --> X=  ', findLostX_Udemy('1X *11 = 121' ))
print('1X0/3 = 50 --> X=  ', findLostX_Udemy('1X0/3 = 50' ))

#%% array rotasyon 
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

#%% array pairs
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


def arrayPairs_Udemy(array):
    
    new =''
    
    for k in range (len(array)):
        
        if k !=0 :
            new += " "
        new += str(array[k]) 
        
        if k%2 == 1 and k != len(array)-1 :
           new += "," 
        
    new = new.split(", ")  
    
    depo = []    
    for c in new:
        
        if c[::-1] not in new :
            for l in c.split():
                depo.append(l)
        elif c == c[::-1] and new.count(c) <2 :
            for l in c.split():
                depo.append(l)
            
    if depo == []:
        return 'ok'
    
    return ','.join(depo)
   
print(arrayPairs_Udemy([5,6,4,5,6,5,3,3,2,2]))

#%%    



















































