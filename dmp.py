# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 08:57:35 2018

@author: Justin Deterding
"""
# IMPORTS ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, trapz 

def Norm(x,y):
    '''
    Returns to normalization coefficient such that trapz(y**2)=1
    Where:
        x: Postion cordinate
        y: Value y(x)
    Returns:
        N: Normilization constant
    '''
    A=trapz(y*y,x=x)
    if A==0: 
        return np.Inf
    else: 
        return np.sqrt(1/A)

def build_wf(h_x,h_y,order):
    '''
    Input:
        y: The half of the wavefunction calculated
        order: The integer order on the wavfunction 
    
    '''
    x=np.append(h_x,-1.0*np.flip(h_x,0))    # Reflect x about 0
    parity = order-(order//2)*2             # Determine parity
    if parity==0:                           # Build even wavefunction
        yr=np.copy(np.flip(h_y,0))          # Reflect wavefunction about zero 
        yr[:,1]=-1.0*yr[:,1]                # Invert wave function
        y=np.append(h_y,yr,0)               # Join the two halfs
    elif parity==1:                         # Build odd wavefunction
        yr=np.copy(np.flip(h_y,0))          # Reflect wavefunction about zero
        yr[:,0]=-1.0*yr[:,0]                # Invert derivative of wavefunction
        y=np.append(h_y,yr,0)               # Join the two halfs
    else:
        print('\a ***Order must be int for build_wf***')
    return Norm(x,y[:,0])*y

def dmp(x,B,d,dR,a,dmp_max=25):
    '''
    Double Morris Potential
    x:  is the position variable (scalar or iterable container of scalars)
    B:  Parameter controling well scale
    d:  well depth
    dR: Seperation between minnimas
            dR<0 -> Classical Transition (No barrier)
            dr>0 -> Possible Tunnaling (barrier)
    a:  Paramiter controling individial well width
    '''
    A = 2*np.exp(-1.0*dR)           # A e**(-dR)
    t1= A**2/2.0*np.cosh(2.0*a*x)   # First term in the potential
    t2= 2*A*np.cosh(a*x)            # Second term in the potential
    p = B*d*(t1-t2)                   # The calculated potential 
    #If we leave the potential unbounded we get overflow errors
    if hasattr(p,'__iter__'):   # Is itterable
        for i in range(len(p)): # Traverse the list and cap if over max_p
          if p[i]>dmp_max:        
            p[i]=dmp_max
    else:
        if p > dmp_max:
            p=dmp_max
    return p

def fdmp(x,t,B,dR,d,a,e):# for Shoot method of solving BVP
    z =x[1]                             #x[1]=dx/dt
    zp=1.0*B*(dmp(t,B,dR,d,a)-e)*x[0]   #zp = (dx/dt)**2
    return np.array([z,zp])

def shoot(e,dmp_args,y0,bc=0,return_type='score'):
    '''
        e: (single bouble value) The eigenvalue of the differential equation
                being solved
        dmp_args: (tuple) list of arguments for the Double Morris potential
                See dmp for explination of arguments
        bc: (0 or 1) 0-> y [end] = 0  condition for a node at center
                     1-> y'[end] = 0  condition no node at center
        return_type: (string) 'score'-> return the score of the integration
                              'wave function'-> return position and derivites
    '''
    args=tuple(list(dmp_args)+[e])      # merge dmp_args with eigenvalue
    y=odeint(fdmp,y0,x,args=args)   # Integrate  
    s=np.log(abs(y[-1,bc]))             # Score the integration based on the 
    if return_type=='score':
        return s            # Return only the score
    else:
        return y            # Return the wavefunction
 
def display(x,y):
        #Display used to debug searching methods
        plt.figure('3pt Ridge Climb');  plt.plot(x,y,'bo')  
        plt.draw();    plt.pause(.01)

def spanning_search(x_span,fun,y_min,disp=False,fun_args=(),n_span=10,I=30):
    '''
    Spanning searchs for a local minumum between a and b. This is done by 
    selceting n_span inearly spaced points between a & b. For each iteration 
    the points one index > and one index < are the new x_span. The process is 
    continued till I iterations is reached or a y<y_min is found.
    Inputs:
        x_span: Contaner of two doubles where the minimum is known to be 
                between the two endpoints (x_span is found using the climb 
                search)
        fun:    A callable function to evaluate the score (y) of the x value. 
                x must be the first argument of the function y=fun(x,*fun_args)
        y_min:  The y value such that if y=fun(x*fun_args)<fun_args then 
                solution is found
        disp:   True-> Display progress of the search on plot as search is done
                False-> Run silently no display
        fun_args: Arguments of the callable scoring function.
        n_span: Number of sub sections to slice the range into for each 
                iteration. NOTE: MUST BE >=4
        I:      Maximum number of iterations to run
    '''
    for i in range(I):
        x = np.linspace(x_span[1],x_span[0],n_span)
        y = [fun(xp,*fun_args) for xp in x]
        if disp: display(x,y)
        k =y.index(min(y))  
        if y[k]< y_min:
            return (x[k],y[k])
        elif k==0 or k==n_span-1:
            print('\a Error minimum should not be at bounds')
            return (x[k],y[k])
        else:
            x_span=[x[k-1],x[k+1]]
    print('\a Minimum y value not acheved in alloted itterations')
    return (x[k],y[k])


def climb(x0,fun,fun_args=(),A=1,NLTF=1,loc_return='min',x_tol=(1e-5,0.1),
          max_iter=100,disp=False):
    '''
    Climb is a searching algorithum that starts at x0 and searches in the +x 
    direction for local minimums and maximums. x is advance inversly 
    proportional to the slope of the function A is the propotionality const.
        x0:         (double) Starting point
        fun:        (Callable function) Function with the form y=fun(x,*fun_args)
        fun_args:   (tuple) Arguments of fun
        NLTF:       (int) Number of Local To Find
        loc_return: (string) 'min'->return NLTF minimums 
                             'max'->return NLTF maximums
                             'both'-> return NLTF minimums and maximums
        x_tol:      (tuple of 2 doubles) x_tol[0]-> minimum x climbing step
                                         x_tol[1]-> maximum x climbing step
        A:          (Double) x[i+1]=x[i]+A*1/m[i-(i-1)]
                             Big A    -> Climb faster
                             Little A -> Climb slower
        max_iter:   (int) Maximum number of iterations to run
        disp:       (bool) Display the y vs.x plot as alg runs.
    '''
    def inv_slope(x,y):
        # Calculate the inverse of the slope. All steps are taken proportional
        # to the invers of the slope.
        m0=min(abs((x[1]-x[0])/(y[1]-y[0])),x_tol[1])
        m1=min(abs((x[2]-x[1])/(y[2]-y[1])),x_tol[1])
        return [m0,m1]         
    # Initilization ----------------------------------------------------------    
    n=0;    l=0                 # Initilize counters    
    loc_max=[];   loc_min=[]    # Initalize containers for locals 
    x=[0,0,x0]                  # Set first 3 points
    y=[0,0,fun(x0,*fun_args)]   # Y(x) values for starting points 
    # Enter main loop --------------------------------------------------------
    while n<max_iter:
        if y[0]>y[1] and y[1]<y[2] and n>0:   # Local minimum was found
            # Store cordinates to left and right of the local minimum 
            loc_min.append((x[0],x[2])); 
            if loc_return=='min' or loc_return=='both': l+=1
            if l==NLTF: break
        elif y[0]<y[1] and y[1]>y[2] and n>0:   # Local maximum was found
            # Store cordinates to left and right of the local minimum 
            loc_max.append((x[0],x[2]));
            if loc_return=='max' or loc_return=='both': l+=1
            if l==NLTF: break
        # x[0], x[1], x[2] need to be repositioned
        x[0]=x[2];  x[1]=x[0]+x_tol[0];   x[2]=x[0]+2*x_tol[0]    
        # Recalculate y[x]
        y=[fun(xp,*fun_args) for xp in x]
        m=inv_slope(x,y)     # Calculate inverse slopes
        if disp:   display(x,y)    # Display
        # While trending downhill or uphill    
        while y[0]>=y[1]>=y[2] or y[0]<=y[1]<=y[2]:    
            #Advance x,y,& m
            x[0]=x[1];  y[0]=y[1];  x[1]=x[2];  y[1]=y[2];  m[0]=m[1]
            x[2]=x[1]+A*m[1]            # linearly project e2    
            y[2]=fun(x[2],*fun_args)    # shoot to calc score and y
            m=inv_slope(x,y)            # calculate new inverse slope
            if disp:   display(x[2],y[2])    # Display
            n+=1
    # Display how the Algorithum terminated
    if n>max_iter: print("Maximum iterations reached")
    else: print("Found %i locals" % (l))
    # Return -----------------------------------------------------------------
    if loc_return == 'min':
        return loc_min
    elif loc_return == 'max':
        return loc_max
    elif loc_return == 'both':
        return loc_min,loc_max
    else:
        print ('\a incorrect loc_return value')
        return None
# End of climbing algorithum -------------------------------------------------

def find_sol(n,ya,x,pot,pot_args,disp=False):
    '''
    find_sol finds the 0 - nth Energy eigenvalue and wavefunction to the 
    schrodinger equation for a fiven potential
    Input:
        n:          (int) n>=0 n is the highest order solution to be found
        ya:         (list dbl) the initial conditions for the solver
        x:          (iterable bdl) the position cordinated to solve over
        pot:        (function) the potential function to solve with
        pat_args:   (tuple) arguments of the potential function
        disp:       (bool) display output True of False
    Output:
        basis:      (list) list of energy eigen values and wavefunctions 
                    [[e0,y0],[e1,y1],[e2,y2],...[en,yn]]
    '''
    p = pot(x,*pot_args)            # Calculate the potential
    e_odd = min(p)                  # Starting energy level for odds
    e_even= min(p)                  # Starting energy level for evens
    basis = []                      # Empty list to store solutions
    # Main loop of fin_sol-----------------------------------------------------
    for i in range(n+1):
        parity = i-(i//2)*2         # Determine parity
        # Set Scoring function based on parity of the nth state ---------------
        if parity == 0:             # Even Parity function
            se = lambda e: shoot(e,pot_args,    # Scoring fun(e) scored on y'[b]
                                 ya,bc=1)       # bc=0 -> score on y[-1,1]
            xe = climb(e_even,          # Energy level to begin search at
                   se,                  # Scoring function
                   NLTF=2,              # Return first local min and max
                   loc_return='both',   # Return mins and max
                   disp=disp,           # Display Ture of false
                   max_iter=300)        # Max iteration depth
            e_even = xe[1][0][1]       # Reset e_start beyond maximum
        elif parity == 1:           # Odd Parity function
            se = lambda e: shoot(e,pot_args,    # Scoring fun(e) scored on y[b]
                                 ya,bc=0)       # bc=0 -> score on y[-1,0]
            xe = climb(e_odd,               # Energy level to begin search at
                       se,                  # Scoring function
                       NLTF=2,              # Return first local min and max
                       loc_return='both',   # Return mins and max
                       disp=disp,           # Display Ture of false
                       max_iter=300)        # Max iteration depth
            e_odd = xe[1][0][1]             # Reset e_start beyond maximum
        else: 
            print( "\a Error in determining parity" )
        # End of setting Scoring function -------------------------------------
        e = spanning_search(xe[0][0],   # Range from climb to span over
                            se,         # Scoring function
                            y_min=-10,  # Minimum y value se much reach to end
                            disp=disp)  # Display progress bool
        y = shoot(e[0],                 # Found eigen energy value
                  pot_args,             # Arguments of the Pot. E function 
                  ya,                   # Left side boundry conditions
                  return_type=1)        # Return wavefunction
        y = build_wf(x,y,i)             # Build full wavefunction
        basis.append([e[0],y])          # Add nth solution to basis
        # End of find_sol main loop -------------------------------------------
    return basis
    
# Main ----------------------------------------------------------------------- 
if __name__ == "__main__":

    ya=np.array([-1e-9,1e-9])       # init. cond. for solver[y[a],y'[b]]
    x=np.linspace(-12,0,1000)       # postion cordinates to solve over
    dmp_args=(1,3,3,.6)             # DMP parameters (B,d,dR,a)
    

    basis = find_sol(6,             # Find 0th-6th energy state
                     ya,x,          # Left boundry condition and position cord.
                     dmp,dmp_args,  # Potential function and arguments
                     disp=False)     # Display progress

    x=np.append(x,-1.0*np.flip(x,0))    # Reflect x about 0
    p=dmp(x,*dmp_args)              # Calculate the Double Morris Potential
    # Plot the wave functions 
    plt.figure('Wave Functions');   n=0
    for base in basis:              
        plt.plot(x,base[1][:,0], label='n='+str(n));   n+=1
    plt.legend()
    # Plot the first derivative of the wavefunction
    plt.figure('d/dx Wavefunctions');   n=0
    for base in basis:              
        plt.plot(x,base[1][:,1], label='n='+str(n));   n+=1
    plt.legend()
    
    # Create the donner and accepter wavefunctions
    accepter =1/np.sqrt(2)*(basis[0][1]+basis[1][1])    # Accepter WF
    donner   =1/np.sqrt(2)*(basis[0][1]-basis[1][1])    # Donner WF
    #plot donnot and accepter probability distributions and potential
    fig1, ax10 = plt.subplots()                 # Two axies energy and  wave fun
    ax10.plot(x,p,'b-')                         # Plot of potential function
    n=0
    for base in basis:              
        plt.plot(x,np.ones(len(x))*base[0],
                 label='n='+str(n))
        n+=1
    plt.legend()

    ax11 = ax10.twinx() 
    ax11.plot(x,accepter[:,0]**2,'r-',          # Plot Accepter wavefunction
              label='Accepter Wavefunction')
    ax11.plot(x,donner[:,0]**2,'g-',            # Plot Donner wavefunction
              label='donner Wavefunction')
    plt.legend()