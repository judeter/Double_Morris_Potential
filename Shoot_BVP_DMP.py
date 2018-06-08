# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:01:19 2018
@author: Justin Deterding
Using Shooting method to sove Double Morris GDE
GDE:
    x''-B(d*P(t)-e)x=0 --> zp[1]=B(d*P(t)-e)z[0]
    Where:
              1 -2*dR                        -dR      
        P(t)= -e     cosh(2*sqrt(B/d)*x)-2*-e   cosh(sqrt(B/d)*x)     
              2             
        dR: Seperation between minnimas
            dR<0 -> Classical Transition (No barrier)
            dr>0 -> Possible Tunnaling (barrier)
        B : Parameter controling well shape
        d : Depth of well
        
        z[0]=x   z[1]=x' 
        zp[0]=x' zp[1]=x''

Known 

B**(-1/2) Has units of meters

"""

from bvp import shoot
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
#Parameters
B =1    # Set B to 1 for convinience (B essential scales the problem)
d =10   # Well depth in multiples of the QHO unit energy level    

dR=1.3   # Well minima seperation parameter
#
def P(t,dR,d):
    a = np.sqrt(B/d)
    A = np.exp(-1.0*dR)
    t1= A**2/2.0*np.cosh(2.0*a*t)
    t2= 2*A*np.cosh(a*t)
    p = d*(t1-t2)
    if hasattr(p,'__iter__'):
        for i in range(len(p)):
          if p[i]>5:
            p[i]=5  
    else:
        if p>5:
            p=5
    return p

##FUNCTIONS FOR THE GENERAL SHOOTING METHOD BVP SOLVER------------------------
def fDMP(x,t,dR,d,e):# for Shoot method of solving BVP
    z =x[1]                         #x[1]=dx/dt
    zp=1.0*B*(P(t,dR,d)-e)*x[0]     #zp = (dx/dt)**2
    return np.array([z,zp])


#Solution Guesses and solution range
x_max=12    #Maximum range for position cordinate (dependent on B)
t = np.linspace(-1.0*x_max,x_max,1000)
#We can make a and b 0 assuming a large enough tollarance on the solver (1e-4)
a=0     #Left boundry value "Guess" 
b=0     #Right boundry value "Guess"
z1=-1e-3    #Note Z1 and z2 cannot be z1=z2 or z1=-z1
z2=-1e-6

score_list =[]
e_list=[]
# The state (0->Ground, 1->First excited state ...)
#e=-3.2876266799 5th bound state 11 peaks
#e=-2.5531251000 6th bound state 12 peaks
#e=-1.7580085860 7th bound state 14 peaks
e=np.linspace(-3,-4,50000)
p_count=0
e_min=[10,0]
for e_ in e:
    
    fdmp = lambda x,t:fDMP(x,t,dR,d,e_)
    
    y_shoot,yp_shoot, found, score = shoot(fdmp,a,b,z1,z2,t,
                                           tol=1e-15, norm_tol=1e-3,
                                           verbose=0)
    if score == 'ERROR':
        print('Error encountered')
    else:
        print("e: %.10f score: %.10f  found: " %(e_,score),end='')
        print(found)
        score_list.append(score)
        e_list.append(e_)
    if found:
        if score < e_min[0]:
            e_min[0]=score
            e_min[1]=e_
        if p_count==0:
            plt.figure(2)
            plt.plot(t,y_shoot,label='shoot')
            plt.plot(t,yp_shoot,label='Dshoot')
            plt.legend()
            p_count=1
    
   
    
    
plt.figure(1)  
plt.plot(e_list,score_list,'bo')
plt.ylim(0,3)

#Plotting---------------------------------------------------------------------
#plt.figure()
#plt.plot(t,P(t,dR,d),label='potential')
#plt.figure()
#if found:
#plt.figure(2)
#plt.plot(t,y_shoot,label='shoot')
#plt.plot(t,yp_shoot,label='Dshoot')
#plt.legend()

 




