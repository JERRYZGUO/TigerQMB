#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:47:14 2019

@author: shirongbao
"""


import numpy as np
import math
import random
import itertools
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time

# difference measure of two curves
def diff(mat1, mat2):
    average = (mat1 + mat2)/2
    
    diff_sqrt = abs(mat1-mat2)
    
    return sum(sum(diff_sqrt/average))

def group(W1, W2):
    W1_Flat = W1.flatten().reshape((N_in+1) * N_hid, 1)
    W2_Flat = W2.flatten().reshape(N_hid, 1)
    
    return np.concatenate((W1_Flat, W2_Flat))

def split(OMEGA):
    W1_flat = OMEGA[: (N_in+1) * N_hid]
    W2_flat = OMEGA[(N_in+1) * N_hid :]
    
    W1 = np.reshape(W1_flat, (N_hid, N_in + 1))
    W2 = np.reshape(W2_flat, (1, N_hid))

    return W1, W2

# Calculating the 
def AF(v,method = 'act1'):  
    if method == 'act1':
       vv = v*v
#       return np.exp(v)/(1 + np.exp(v)) #Sigmoid (logistic) function
       return 1/2 +v*(1/4-vv*(1/48-vv*(1/480-vv*(17/80640-vv*31/1451520))))   
    elif method == 'act2':
       return v/(1+np.abs(v))
    elif method == 'act3':
       return np.log(np.cosh(v))
    elif method == 'act4':
       return v
    elif method == 'act5':
#       return np.exp(v)
       vv = v*v
       return 1+vv*(1/2+vv*(1/24+vv*(1/720+vv/40320)))+v*(1+vv*(1/6+vv*(1/120+vv/5040)))   
    elif method == 'act6':
       vv = v*v
#       return np.cosh(v)
       return 1+vv*(1/2+vv*(1/24+vv*(1/720+vv/40320)))      
    elif method == 'act7':
       vv = v*v
       return np.tanh(v)
#       return v*(1-vv*(1/3-vv*2/15))
#       return v*(1-vv*(1/3-vv*(2/15-17/315*vv)))
#       return v*(1-vv*(1/3-vv*(2/15-vv*(17/315-62/2835*vv)))) 
    elif method == 'act8':
       vv = v**2
#       return vv*(1/2-vv*(1/12-vv/45)) # Maclaurin series of ln(cosh x)
#       return vv*(1/2-vv*(1/12-vv*(1/45-17/2520*vv)))
       return vv*(1/2-vv*(1/12-vv*(1/45-vv*(17/2520-62/28350*vv))))     

def dAF(v, method = 'act1'):
    if method == 'act1':
       vv = v*v
#       return np.exp(v)/((1 + np.exp(v))**2)
       return 1/4 - vv*(1/16-vv*(1/96-vv*(17/11520-vv*31/161280))) 
    elif method == 'act2':
       return (1 + 0.5 * np.abs(v))/((1 + np.abs(v))**2)
    elif method == 'act3':
       return np.tanh(v)
    elif method == 'act4':
       return 1
    elif method == 'act5':
#       return np.exp(v)
       vv = v*v
       return 1+vv*(1/2+vv*(1/24+vv*(1/720+vv/40320)))+v*(1+vv*(1/6+vv*(1/120+vv/5040)))     
    elif method == 'act6':
       vv = v*v
#       return np.sinh(v)
       return v*(1+vv*(1/6+vv*(1/120+vv/5040)))   
    elif method == 'act7':
       vv = v*v
       return 1.0-(np.tanh(v))**2
#       return 1-vv*(1-vv*2/3)
#       return 1-vv*(1-vv*(2/3-17/45*vv))
#       return 1-vv*(1-vv*(2/3-vv*(17/45-62/315*vv)))    
    elif method == 'act8':
       vv = v**2
#       return v*(1-vv*(1/3-vv*2/15)) # Maclaurin series of tanh x
#       return v*(1-vv*(1/3-vv*(2/15-17/315*vv)))
       return v*(1-vv*(1/3-vv*(2/15-vv*(17/315-62/2835*vv))))   
    
def PSI(OMEGA, v, method = 'linear'):
    
    W1, W2 = split(OMEGA)
    v_1 = np.append(np.array([1]), v).reshape(N_in+1, 1)  
    v_hid = np.dot(W1, v_1)
    v_hid_1 = AF(v_hid).reshape(N_hid, 1)
    if method =='exponential':
         return np.exp(np.dot(W2, v_hid_1)[0, 0]), v_hid
    elif method == 'linear':
         return np.dot(W2, v_hid_1)[0, 0], v_hid

# the RBM derivative of psi respect to a, b, and W
def dpsi(OMEGA, v, method = 'linear'):
   
    v_1 = np.append(np.array([1]), v).reshape(N_in+1, 1)
    W1, W2 = split(OMEGA)
    
    # Function values
    psi, v_hid = PSI(OMEGA, v)
    
    v_hid_1 = AF(v_hid).reshape(N_hid, 1)
    if method == 'exponential':
         dpsi_W2 = v_hid_1.T * psi
         dpsi_W1 = np.dot((W2.T * dAF(v_hid)), v_1.T) * psi
    elif method == 'linear':
         dpsi_W2 = v_hid_1.T
         dpsi_W1 = np.dot((W2.T * dAF(v_hid)), v_1.T)
    
    return group(dpsi_W1, dpsi_W2)

# the exact method derivative of psi respect to a, b, and W
def exact_dpsi(t, psi):
    return - complex(0.0, 1.0) * np.dot(H(t), psi)
    
# the matrix form of hamiltonian at time t
def H(t):
    if t <0:
#        B1 = 0
        B1 = CTF0
        J1 = J0
    else:
#        B1 = Amp * np.power(np.sin((math.pi * t)/T), 2.0) * np.cos(omega * 3/2 * t)
        B1 = CTF + Amp * np.power(np.sin((math.pi * t)/T), 2.0) * np.cos(omega * t)
#        B1 = Amp  
        J1 = J

    H1_mat = (B0 * Pauli_Z + B1 * Pauli_X)
    H1 = np.array([[0]])
    for i in range(N_in):
        H1_temp = np.array([[1]])
        for j in range(N_in):
            if j != i:
                H1_temp = np.kron(H1_temp, id_mat)
            else:
                H1_temp = np.kron(H1_temp, H1_mat)
        H1 = H1 + H1_temp
    
    H2 = np.array([[0]])
    if N_in >= 1:
        H2_mat = J1 * np.kron(Pauli_Z, Pauli_Z)
        for i in range(N_in-1):
            H2_temp = np.array([[1]])
            for j in range(N_in-1):
                if j != i:
                    H2_temp = np.kron(H2_temp, id_mat)
                else:
                    H2_temp = np.kron(H2_temp, H2_mat)
            H2 = H2 + H2_temp
        
    return H1 + H2 

# hermitian of matrix of vector
def herm(v):
    return np.conjugate(v.T)
method1 = 'SR0 e-15'
# output dOMEGA/dt. pinv for pseudoinverse, SR for stochastic reconfiguration.
def dOMEGA(OMEGA, t, method = 'SR0'):
    A = np.zeros((len(States), OMEGA.size), complex)
    psi_vector = np.zeros((len(States), 1), complex)
    for i in range(len(States)):
        cur_state = States[i]
        
        v = cur_state.T.reshape((N_in, 1))
        A[i] = dpsi(OMEGA, v).T
        
        psi, v_hid = PSI(OMEGA, v)
        psi_vector[i] = psi
    
    if method == 'SR1':
        V, sig, Uh = LA.svd(np.conjugate(A.T))
    
        if len(States) >= OMEGA.size:
            sig = np.dot(np.eye(len(States), OMEGA.size), np.diag(sig)).T
        else:
            sig = np.dot(np.diag(sig), np.eye(len(States), OMEGA.size)).T
        Q = np.dot(sig, sig.T)
#        - np.dot(np.dot(sig, np.dot(Uh, psi_vector)), np.dot(np.dot(herm(psi_vector),herm(Uh)),sig.T))
        
        F = np.dot(sig, np.dot(Uh, np.dot(H(t), psi_vector))) - np.dot(herm(psi_vector), np.dot(H(t), psi_vector))/np.dot(herm(psi_vector),psi_vector) * np.dot(sig, np.dot(Uh, psi_vector))
        
#        return -complex(0, 1) * np.dot(V, np.dot(LA.pinv(Q, hermitian = True), F))
        return -complex(0, 1) * np.dot(V, np.dot(LA.pinv(Q, rcond=1.0e-6, hermitian = True), F))    

    elif method == 'SR':
        V, sig, Uh = LA.svd(np.conjugate(A.T))
    
        if len(States) >= OMEGA.size:
            sig = np.dot(np.eye(len(States), OMEGA.size), np.diag(sig)).T
        else:
            sig = np.dot(np.diag(sig), np.eye(len(States), OMEGA.size)).T
        Q = np.dot(sig, sig.T)- np.dot(np.dot(sig, np.dot(Uh, psi_vector)), np.dot(np.dot(herm(psi_vector),herm(Uh)),sig.T))
        
        F = np.dot(sig, np.dot(Uh, np.dot(H(t), psi_vector))) - (np.dot(herm(psi_vector), np.dot(H(t), psi_vector)))/(np.dot(herm(psi_vector),psi_vector)) * np.dot(sig, np.dot(Uh, psi_vector))
        
#        return -complex(0, 1) * np.dot(V, np.dot(LA.pinv(Q, hermitian = True), F))
        return -complex(0, 1) * np.dot(V, np.dot(LA.pinv(Q, rcond=1.0e-15, hermitian = True), F))  
    
    elif method == 'SR0':
        V, sig, Uh = LA.svd(np.conjugate(A.T))
    
        if len(States) >= OMEGA.size:
            sig = np.dot(np.eye(len(States), OMEGA.size), np.diag(sig)).T
        else:
            sig = np.dot(np.diag(sig), np.eye(len(States), OMEGA.size)).T
        Q = np.dot(sig, sig.T)
#        - np.dot(np.dot(sig, np.dot(Uh, psi_vector)), np.dot(np.dot(herm(psi_vector),herm(Uh)),sig.T))
        
        F = np.dot(sig, np.dot(Uh, np.dot(H(t), psi_vector)))
        
#        - np.dot(herm(psi_vector), np.dot(H(t), psi_vector))/np.dot(herm(psi_vector),psi_vector) * np.dot(sig, np.dot(Uh, psi_vector))
        
#        return -complex(0, 1) * np.dot(V, np.dot(LA.pinv(Q, hermitian = True), F))
        return -complex(0, 1) * np.dot(V, np.dot(LA.pinv(Q,rcond=1.0e-15, hermitian = True), F))    
    


    elif method == 'pinv':
        S = np.dot(np.conjugate(A.T), A)
        F = np.dot(np.conjugate(A.T), np.dot(H(t), psi_vector))
#        return -complex(0,1) * np.dot(LA.pinv(S, hermitian = True), F)
        return -complex(0,1) * np.dot(LA.pinv(S, rcond=1.0e-12, hermitian = True), F)

    elif method == 'pinv1':
        SS=np.dot(np.conjugate(A.T),psi_vector)
        SN=(np.dot(herm(psi_vector),psi_vector))
        S = np.dot(np.conjugate(A.T), A)-np.dot(SS,np.conjugate(SS.T))/SN
        F = np.dot(np.conjugate(A.T), np.dot(H(t), psi_vector))-SS*np.dot(herm(psi_vector), np.dot(H(t), psi_vector))/SN
#        return -complex(0,1) * np.dot(LA.pinv(S, hermitian = True), F)
        return -complex(0,1) * np.dot(LA.pinv(S, rcond=1.0e-6, hermitian = True), F)
    
    

def fix(OMEGA):
    for i in range((N_in+1) * N_hid, (N_in +1) * N_hid + N_hid):
        OMEGA[i] = complex(1, 0)
    return OMEGA
method2 = 'pinv1 e-06'
# Evolution of OMEGA for initialization. Imaginary time evolution
def evol(OMEGA, stepsize, perc):
    stepsize =  stepsize * np.exp(perc * 4)
    return OMEGA + stepsize * complex(0, -1) * dOMEGA(OMEGA, -1, method = 'pinv1')

# Calculate the local energy
def loc_E(OMEGA, t):
    psi_vector = np.zeros((len(States), 1), complex)
    for i in range(len(States)):
        cur_state = States[i]
        
        v = cur_state.T.reshape((N_in, 1))
        
        psi, v_hid = PSI(OMEGA, v)
        
        psi_vector[i] = psi
    
    return sum(np.dot(herm(psi_vector), np.dot(H(-1), psi_vector)))/sum(np.dot(herm(psi_vector), psi_vector))

# RK integrator 4th order for RBM
def RK(t, OMEGA, step, Nsub): 
    # step height h 
    h = step/Nsub
    t1 = t
    for i in range(Nsub):
       k1 = h * dOMEGA(OMEGA, t1)
       k2 = h * dOMEGA(OMEGA + 0.5 * k1, t1 + 0.5 * h)
       k3 = h * dOMEGA(OMEGA + 0.5 * k2, t1 + 0.5 * h)
       k4 = h * dOMEGA(OMEGA + k3, t1 + h)
    
    # Update next value of OMEGA 
       OMEGA = OMEGA + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
       t1 = t1 + h
    # Update next value of t
    t = t1 
    return OMEGA, t

def Heun(t, OMEGA, step, Nsub): 
    h = step/Nsub
    t1 = t
    # step height h
    for i in range(Nsub):
        k1 = h * dOMEGA(OMEGA, t1)
        k2 = h * dOMEGA(OMEGA + k1, t1 + h)

    # Update next value of OMEGA 
        OMEGA = OMEGA + (1.0 / 2.0)*(k1 + k2) 
 
    # Update next value of t
        t1 = t1 + h
    t = t1    
    return OMEGA, t

def Euler(t, OMEGA, step, Nsub): 
    h = step/Nsub
    t1 = t
    # step height h
    for i in range(Nsub):
        k1 = h * dOMEGA(OMEGA, t1)
    # Update next value of OMEGA 
        OMEGA = OMEGA + k1 
        t1 = t1 + h 
    # Update next value of t
 
    t = t1    
    return OMEGA, t    

def RK4(t, OMEGA, step): 
    # step height h 
    h = step
    k1 = h * dOMEGA(OMEGA, t)
    k2 = h * dOMEGA(OMEGA + 0.5 * k1, t + 0.5 * h)
    k3 = h * dOMEGA(OMEGA + 0.5 * k2, t + 0.5 * h)
    k4 = h * dOMEGA(OMEGA + k3, t + h)
    
    # Update next value of OMEGA 
    OMEGA = OMEGA + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
    
    # Update next value of t
    t = t + h 
    return OMEGA, t

# RK integrator 4th order for exact method
def exact_RK(t, step, psi): 
        # step height h 
    h = step
    k1 = h * exact_dpsi(t, psi)
    k2 = h * exact_dpsi(t + 0.5 * h, psi + 0.5 * k1)
    k3 = h * exact_dpsi(t + 0.5 * h, psi + 0.5 * k2)
    k4 = h * exact_dpsi(t + h, psi + k3)
      
    # Update next value of OMEGA 
    psi  = psi + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
      
    # Update next value of t
    t = t + h 
    return psi, t

# Pauli matrix for different dimensions
def Pauli(name = 'Z'):
    if name == 'Z':
        pauli = Pauli_Z
    elif name == 'Y':
        pauli = Pauli_Y
    elif name == 'X':
        pauli = Pauli_X
    
    pauli_final = np.array([[0]])
    for i in range(N_in):
        pauli_temp = np.array([[1]])
        for j in range(N_in):
            if j != i:
                pauli_temp = np.kron(pauli_temp, id_mat)
            else:
                pauli_temp = np.kron(pauli_temp, pauli)
        pauli_final = pauli_final + pauli_temp
    return pauli_final

# expected value of Pauli matrices in x, y, z directions
def exp_Spin(psi_track):
    psi_track = psi_track.T
    exp_track_x = []
    exp_track_y = []
    exp_track_z = []
    
    for i in range(N_Steps):
        exp_track_x.append(np.dot(herm(psi_track[i]), np.dot(Pauli('X'), psi_track[i]))/np.dot(herm(psi_track[i]), psi_track[i]))
        exp_track_y.append(np.dot(herm(psi_track[i]), np.dot(Pauli('Y'), psi_track[i]))/np.dot(herm(psi_track[i]), psi_track[i]))
        exp_track_z.append(np.dot(herm(psi_track[i]), np.dot(Pauli('Z'), psi_track[i]))/np.dot(herm(psi_track[i]), psi_track[i]))
        
    return exp_track_x, exp_track_y, exp_track_z


# Constant matrices
Pauli_X = np.array([[0, 1], [1, 0]])
Pauli_Z = np.array([[-1, 0], [0, 1]])
Pauli_Y = complex(0, 1) * np.array([[0, 1], [-1, 0]])
id_mat = np.array([[1,0],[0,1]])

# Initial parameters
N_in = 5
N_hid = 5
B0 = 0.0 #1.0 #1.0 #1.0 #0.9 #1.0 # constant longitudinal field
CTF0 = 0.5 # constant Transverse field for the initial state Hamiltonian
CTF = 0.5 # constant transverse field for the field-free Hamiltonian
J0 = 1.0  # 0.0 uncoupled # J coupled
J =  1.0  # spin-spin coupling
Amp = 1.0 #0.2 #0.2 #1.0 $ transverse control field amplitue 
#Amp = 0.0 # for quenching unitary dynamics CTF0 --> CTF
print('Longitudinal field B0 = ', B0) 
print('spin-spin coulping J0, J = ', J0, J) 
print('Transverse field CTF0, CFT, Amp = ', CTF0, CTF, Amp)
N_iter = 1000 #1000
stepsize = 0.001 #0.001

N_Steps = 1000

Nsub = 10
scheme_RBM = 'Heun' #'Euler' #'Heun'#'RK'

random.seed(1)

omega = 2.0*J #- (0.1 * random.random())
#omega = (2.0*B0) # - (0.1 * random.random())
T = 5.0 * ((2.0 * math.pi)/omega) #(2*B0)) #omega))
step = T/N_Steps
N_state = np.power(2, N_in)
N_parameter = (N_in +1) * N_hid + N_hid
print('Transverse field frequency = ', omega) 
print('Pulse length, control field cycle = ', T, (2.0 * math.pi)/(omega)) 
print('N_in, N_hid, N_state, N_parameter, N_Steps, Nsub = ', N_in, N_hid, N_state, N_parameter, N_Steps, Nsub)


# Randomly initializae OMEGA
OMEGA = np.zeros(((N_in+1) * N_hid+N_hid, 1), complex)
#random.seed(1)
for i in range((N_in +1)* N_hid + N_hid):
    OMEGA[i] = complex(2*random.random()-1, 2*random.random()-1)
OMEGA_random = OMEGA

# Initialize all possible states
#comb = itertools.combinations_with_replacement([0,1], N_in)
comb = itertools.combinations_with_replacement([-1,1], N_in)
States = []
for x in comb:
	perm = list(set(itertools.permutations(x, N_in)))
	for y in perm:
		States.append(y)
States = np.asarray(States)

# Calculate initial random wave function according to the randomly initialized OEMGA
psi_random = []
for i in range(N_state):
    psi, ratio = PSI(OMEGA, States[i].T.reshape((N_in, 1)))
    psi_random.append(psi)

# --------------------------- Optimization/Initialization ----------------------------------
start_time = time.time()

# tracker though the initialization
OMEGA_i_track = []
iter_track = []
E_track = []

# Evolve OMEGA and record
for i in range(N_iter):
    if i%int(N_iter/10) == 0:
        print("Initialization: %.1f" % (i/N_iter*100), '%') 
    OMEGA_i_track.append(OMEGA)
    OMEGA = evol(OMEGA, stepsize, i/N_iter)   
    iter_track.append(i)
    E_track.append(loc_E(OMEGA, 0))
#for i in range(1):    
print('local energy =', E_track[0], E_track[N_iter-1])


omega1_i_track= []  
omega2_i_track= []   
for i in range(N_parameter):
    cur_omega1 = []
    cur_omega2 = []
    for OMEGA in OMEGA_i_track:
        comega = OMEGA[i]
        omega1 = np.real(comega) 
        omega2 = np.imag(comega)
        cur_omega1.append(omega1)
        cur_omega2.append(omega2)
    omega1_i_track.append(cur_omega1) 
    omega2_i_track.append(cur_omega2) 

# Calculate probability density of states according to OMEGA at each time instants
PD_i_track = []
for i in range(N_state):
    cur_PD = []
    for OMEGA in OMEGA_i_track:
        psi, ratio = PSI(OMEGA, States[i].T.reshape((N_in, 1)))
        cur_PD.append(abs(psi)**2)
    PD_i_track.append(cur_PD)

# Normalize the probability density track according to the sum at t = 0
PD_i_track = PD_i_track/sum(temp[0] for temp in PD_i_track)

# track of sum of normalized PD track
PD_i_sum = []
for i in range(len(OMEGA_i_track)):
    PD_i_sum.append(sum(PD_i_track[:, i]))
    
#plot initialization dada 
 
fig, axs=plt.subplots(2, figsize=(7,7), sharex=True)
#fig.suptitle('Initialization, Nin = {:d}, Nhid = {:d}, stepsize = {:f}'.format(N_in, N_hid, stepsize))   
# plot the local energy throughout iterations
#for i in range(len(E_track)):
axs[0].plot(iter_track, E_track)
#plt.legend()
#axs[0].set_xlabel('iterations')
axs[0].set_ylabel('local energy')
axs[0].set_xlim(0, N_iter)
axs[0].set_title("Initialization, Nin = {:d}, Nhid = {:d}, N = {:d}, M = {:d},  B0 = {:f}, J = {:f}, stepsize = {:f} ".format(N_in, N_hid, N_state, N_parameter, B0, J, stepsize), fontsize=8)
#plt.show()

# plot PD and their sum
for x in range(N_state):
    axs[1].plot(iter_track, PD_i_track[x]/PD_i_sum)
#plt.plot(iter_track, PD_i_sum, label='Sum')
#plt.legend()
#axs[1].set_xlabel('iterations')
axs[1].set_ylabel('state probability')
axs[1].set_xlim(0, N_iter)
#axs[1].set_ylim(0.0, 1.0)
#for ax in axs.flat:
axs[1].set(xlabel='iterations')
#axs[1].set_title("Initialization, Nin = {:d}, Nhid = {:d}, stepsize = {:f}".format(N_in, N_hid, stepsize))
plt.show()
fig.savefig('FNNfig1.pdf')

fig, axs=plt.subplots(3,2, figsize=(9,7.5), sharex=True) 
fig.suptitle('FNN imaginary-time parameters, Nin = {:d}, Nhid = {:d}, N = {:d}, M = {:d}, N_iter = {:d}, B0 = {:f}, J = {:f}'.format(N_in, N_hid, N_state, N_parameter, N_iter, B0,J), fontsize=8) 
#plot ANN parameters as a function of iterations     
for x in range(N_in*N_hid): #N_parameter):
    axs[0,0].plot(iter_track, omega1_i_track[x])
#plt.xlabel('iterations')
axs[0,0].set_ylabel('real part W1')
axs[0,0].set_xlim(0, N_iter)
#plt.ylim(0.0, 1.3)
#axs[0,0].set_title("RBM parameters a, Nin = {:d}, Nhid = {:d}, N_iter = {:d}".format(N_in, N_hid, N_iter))
#plt.show()    
for x in range(N_in*N_hid): #N_parameter):
    axs[0,1].plot(iter_track, omega2_i_track[x])
#plt.legend()
#axs[1].set_xlabel('iterations')
axs[0,1].set_ylabel('imag part W1')
axs[0,1].set_xlim(0, N_iter)
#plt.ylim(0.0, 1.3)
#plt.title("RBM Method, Nin = {:d}, Nhid = {:d}, N_iter = {:d}".format(N_in, N_hid, N_iter))
#for ax in axs.flat:
#axs[0,1].set(xlabel='iterations')
#plt.show()
#fig.savefig('RBMfig2.pdf')

#fig, axs=plt.subplots(2, figsize=(7,7)) 
#fig.suptitle('RBM parameters b, Nin = {:d}, Nhid = {:d}, N_iter = {:d}'.format(N_in, N_hid, N_iter)) 
for x in range(N_in*N_hid,(N_in+1)*N_hid):
    axs[1,0].plot(iter_track, omega1_i_track[x])
#plt.legend()
#plt.xlabel('iterations')
axs[1,0].set_ylabel('real part b')
axs[1,0].set_xlim(0, N_iter)
#plt.ylim(0.0, 1.3)
#axs[1,0].set_title("RBM parameters b, Nin = {:d}, Nhid = {:d}, N_iter = {:d}".format(N_in, N_hid, N_iter))
#plt.show()

for x in range(N_in*N_hid,(N_in+1)* N_hid):
    axs[1,1].plot(iter_track, omega2_i_track[x])
#plt.legend()
#axs[1].set_xlabel('iterations')
axs[1,1].set_ylabel('imag part b')
axs[1,1].set_xlim(0, N_iter)
#plt.ylim(0.0, 1.3)
#plt.title("RBM Method, Nin = {:d}, Nhid = {:d}, N_iter = {:d}".format(N_in, N_hid, N_iter))
#for ax in axs.flat:
#axs[1,1].set(xlabel='iterations')
#plt.show()
#fig.savefig('RBMfig3.pdf')

#fig, axs=plt.subplots(2, figsize=(7,7)) 
#fig.suptitle('RBM parameters W, Nin = {:d}, Nhid = {:d}, N_iter = {:d}'.format(N_in, N_hid, N_iter)) 
for x in range((N_in+1)*N_hid,(N_in +1)* N_hid +  N_hid):
    axs[2,0].plot(iter_track, omega1_i_track[x])
#plt.legend()
#axs[0].set_xlabel('iterations')
axs[2,0].set_ylabel('real part W2')
axs[2,0].set_xlim(0, N_iter)
#plt.ylim(0.0, 1.3)
axs[2,0].set(xlabel='iterations')
#axs[2,0].set_title("RBM parameters W, Nin = {:d}, Nhid = {:d}, N_iter = {:d}".format(N_in, N_hid, N_iter))
#plt.show()

for x in range((N_in +1)* N_hid,(N_in +1)* N_hid + N_hid):
    axs[2,1].plot(iter_track, omega2_i_track[x])
#plt.legend()
#axs[1].set_xlabel('iterations')
axs[2,1].set_ylabel('imag part W2')
axs[2,1].set_xlim(0, N_iter)
#plt.ylim(0.0, 1.3)
#plt.title("RBM Method, Nin = {:d}, Nhid = {:d}, N_iter = {:d}".format(N_in, N_hid, N_iter))
#for ax in axs.flat:
axs[2,1].set(xlabel='iterations')
plt.show()    
fig.savefig('FNNfig2.pdf')    
    
# Final OMEGA and wave functions as initial parameters for RBM method and exact method
OMEGA_initial = OMEGA_i_track[-1]
psi_initial = []
for i in range(N_state):
#    psi, ratio = PSI(np.complex(0.5,0.0)*(OMEGA_initial+np.conjugate(OMEGA_initial)), States[i].T.reshape((N_in, 1)))
    psi, ratio = PSI(OMEGA_initial, States[i].T.reshape((N_in, 1)))
    psi_initial.append(psi)

RBM_start = time.time()
print("--- Initialization %s seconds ---" % (RBM_start - start_time))
# --------------------------- RBM METHOD ----------------------------------

# track
#OMEGA = np.complex(0.5,0.0)*(OMEGA_initial + np.conjugate(OMEGA_initial)) 
OMEGA = OMEGA_initial
OMEGA_track = []
t_track = []
B1_track = []
t = 0

# RBM evolution of OMEGA
for i in range(N_Steps):
    if i%int(N_Steps/10) == 0:
        print("FNN: %.1f" % (i/N_Steps*100), '%')
    OMEGA_track.append(OMEGA)
    t_track.append(t)
    B1_track.append(Amp * np.power(np.sin((math.pi * t)/T), 2.0) * np.cos(omega * t))
#    B1_track.append(Amp /2 )    
#    OMEGA, t = RK(t, OMEGA, step)
    if scheme_RBM == 'RK':
       OMEGA, t = RK(t, OMEGA, step, Nsub)
    elif scheme_RBM == 'Heun':   
       OMEGA, t = Heun(t, OMEGA, step, Nsub)
    elif scheme_RBM == 'Euler':   
       OMEGA, t = Euler(t, OMEGA, step, Nsub)


# Calculate probability density and wave functions of states according to OMEGA at each time instants
PD_RBM_track = []
psi_RBM_track = []
for i in range(N_state):
    cur_psi = []
    cur_PD = []
    for OMEGA in OMEGA_track:
        psi, ratio = PSI(OMEGA, States[i].T.reshape((N_in, 1)))
        cur_psi.append(psi)
        cur_PD.append(abs(psi)**2)
    psi_RBM_track.append(cur_psi)
    PD_RBM_track.append(cur_PD)
    
omega1_RBM_track= []  
omega2_RBM_track= []   
for i in range(N_parameter):
    cur_omega1 = []
    cur_omega2 = []
    for OMEGA in OMEGA_track:
        comega = OMEGA[i]
        omega1 = np.real(comega) 
        omega2 = np.imag(comega)
        cur_omega1.append(omega1)
        cur_omega2.append(omega2)
    omega1_RBM_track.append(cur_omega1) 
    omega2_RBM_track.append(cur_omega2) 

# Normalization
PD_RBM_track = PD_RBM_track/sum(temp[0] for temp in PD_RBM_track)

# track of probability density sum
PD_RBM_sum = []
for i in range(len(t_track)):
    PD_RBM_sum.append(sum(PD_RBM_track[:, i]))
    

#plot
fig, axs=plt.subplots(3,2, figsize=(9,7.5), sharex=True) 
fig.suptitle('FNN real-time parameters, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}, Amp = {:f}, omega = {:f}, T = {:f}'.format(N_in, N_hid, N_Steps,Amp,omega,T), fontsize=8) 
for x in range(N_in*N_hid): #N_parameter):
    axs[0,0].plot(t_track, omega1_RBM_track[x])
#plt.legend()
#plt.xlabel('time')
axs[0,0].set_ylabel('real part W1')
axs[0,0].set_xlim(0, T)
#plt.ylim(0.0, 1.3)
#axs[0,0].set_title("RBM parameters a, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}".format(N_in, N_hid, N_Steps))
#plt.show()

for x in range(N_in*N_hid): #N_parameter):
    axs[0,1].plot(t_track, omega2_RBM_track[x])
#plt.legend()
#axs[0,1].set_xlabel('time')
axs[0,1].set_ylabel('imag part W1')
axs[0,1].set_xlim(0, T)
#plt.ylim(0.0, 1.3)
#plt.title("RBM Method, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}".format(N_in, N_hid, N_Steps))
#for ax in axs.flat:
#axs[0,1].set(xlabel='time')
#plt.show()
#fig.savefig('RBMfig5.pdf')

#fig, axs=plt.subplots(2, figsize=(7,7)) 
#fig.suptitle('RBM parameters b, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}'.format(N_in, N_hid, N_Steps)) 
for x in range(N_in*N_hid,(N_in +1)* N_hid):
    axs[1,0].plot(t_track, omega1_RBM_track[x])
#plt.legend()
#plt.xlabel('time')
axs[1,0].set_ylabel('real part b1')
axs[1,0].set_xlim(0, T)
#plt.ylim(0.0, 1.3)
#axs[1,0].set_title("RBM parameters b, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}".format(N_in, N_hid, N_Steps))
#plt.show()

for x in range(N_in*N_hid,(N_in +1)* N_hid):
    axs[1,1].plot(t_track, omega2_RBM_track[x])
#plt.legend()
#plt.xlabel('time')
axs[1,1].set_ylabel('imag part b1')
axs[1,1].set_xlim(0, T)
#plt.ylim(0.0, 1.3)
#plt.title("RBM Method, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}".format(N_in, N_hid, N_Steps))
#for ax in axs.flat:
#axs[1,1].set(xlabel='time')
#plt.show()
#fig, axs=plt.subplots(2, figsize=(7,7)) 

#fig.suptitle('RBM parameters W, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}'.format(N_in, N_hid, N_Steps)) 
for x in range((N_in +1)* N_hid,(N_in +1)* N_hid + N_hid):
    axs[2,0].plot(t_track, omega1_RBM_track[x])
#plt.legend()
#plt.xlabel('time')
axs[2,0].set_ylabel('real part W2')
axs[2,0].set_xlim(0, T)
#plt.ylim(0.0, 1.3)
axs[2,0].set(xlabel='time')
#axs[0].set_title("RBM parameters W, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}".format(N_in, N_hid, N_Steps))
#plt.show()

for x in range((N_in +1)* N_hid,(N_in +1)* N_hid + N_hid):
    axs[2,1].plot(t_track, omega2_RBM_track[x])
#plt.legend()
#plt.xlabel('time')
axs[2,1].set_ylabel('imag part W2')
axs[2,1].set_xlim(0, T)
#plt.ylim(0.0, 1.3)
#plt.title("RBM Method, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}".format(N_in, N_hid, N_Steps))
#for ax in axs.flat:
axs[2,1].set(xlabel='time')
plt.show()
fig.savefig('FNNfig3.pdf')

exact_start = time.time()
print("--- FNN %s seconds ---" % (exact_start - RBM_start))
# --------------------------- EXACT METHOD ----------------------------------

# Initial wave function the same
psi = psi_initial 
t = 0

# tracker
psi_exact_track = []
PD_exact_track = []

# Evolve OMEGA for exact method
for i in range(N_Steps):
    if i%int(N_Steps/10) == 0:
        print("Exact: %.1f" % (i/N_Steps*100), '%')
    psi, t = exact_RK(t, step, psi)
    psi_exact_track.append(psi)
    PD_exact_track.append(np.abs(psi)**2)

# Normalization of PD track
psi_exact_track = np.asarray(psi_exact_track).T.tolist()
PD_exact_track = np.asarray(PD_exact_track).T
PD_exact_track = (PD_exact_track/sum(PD_exact_track[:, 0])).tolist()
    
# tracker for sum
PD_exact_sum = []
for i in range(N_Steps):
    PD_exact_sum.append(sum(temp[i] for temp in PD_exact_track))
    
#plot time-dependent transverse field

#fig, axs=plt.subplots(2, figsize=(7,7))  
fig, axs=plt.subplots(3, figsize=(7,7.5), sharex=True)     
axs[0].plot(t_track, B1_track, label='transverse field')
plt.legend()
#axs[0].set_xlabel('time')
axs[0].set_ylabel('Transverse field')
axs[0].set_xlim(0, T)
axs[0].set_ylim(-Amp,Amp)
axs[0].set_title("Final results, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}, Amp = {:f}, omega = {:f}, T={:f}".format(N_in, N_hid, N_Steps, Amp, omega,T), fontsize=8)
#plt.show()
#plt.savefig('RBMfig8.pdf')
    
# plot

#fig.suptitle('RBM method, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}'.format(N_in, N_hid, N_Steps)) 
for x in range(N_state):
    axs[1].plot(t_track, PD_RBM_track[x]/PD_RBM_sum)
#plt.plot(t_track, PD_RBM_sum, label='Sum')
#plt.legend()
#plt.xlabel('time')
axs[1].set_ylabel('probability FNN')
axs[1].set_xlim(0, T)
#axs[1].set_ylim(0.0, 1.0)
#axs[1].set_title("RBM Method, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}".format(N_in, N_hid, N_Steps))
#plt.show()
    
    
# plot
for x in range(N_state):
    axs[2].plot(t_track, PD_exact_track[x])
#plt.plot(t_track, PD_exact_sum, label='Sum')
#plt.legend()
#plt.xlabel('time')
axs[2].set_ylabel('probability exact')
axs[2].set_xlim(0, T)
#axs[2].set_ylim(0.0, 1.0)
#plt.title("Exact Method, Nin = {:d}, Nsteps = {:d}".format(N_in, N_Steps))
#for ax in axs.flat:
axs[2].set(xlabel='time')
plt.show()
fig.savefig('FNNfig4.pdf')

Analysis_start = time.time()
print("--- Exact %s seconds ---" % (Analysis_start - exact_start))
# --------------------------- Analysis ----------------------------------


print('Difference measure:', diff(np.asarray(PD_RBM_track), np.asarray(PD_exact_track))/(N_state * N_Steps))

# expected value for pauli XYZ matrices for RBM and exact methods
exp_x_RBM, exp_y_RBM, exp_z_RBM = exp_Spin(np.asarray(psi_RBM_track))
exp_x_exact, exp_y_exact, exp_z_exact = exp_Spin(np.asarray(psi_exact_track))

fig, axs=plt.subplots(3, figsize=(7,7.5), sharex=True) 
#fig.suptitle('Pauli X, Y, Z, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}'.format(N_in, N_hid, N_Steps)) 
# plot results
axs[0].plot(t_track, exp_x_RBM, label='FNN')
axs[0].plot(t_track, exp_x_exact, label='exact')
axs[0].legend()
axs[0].set_xlim(0,T)
#plt.title('Pauli X')
#plt.xlabel('time')
axs[0].set_ylabel('<X>')
axs[0].set_title('Pauli X, Y, Z, Nin = {:d}, Nhid = {:d}, Nsteps = {:d}, Amp = {:f}, omega = {:f}, T = {:f}'.format(N_in, N_hid, N_Steps, Amp, omega, T), fontsize=8) 
#plt.show()

axs[1].plot(t_track, exp_y_RBM, label='FNN')
axs[1].plot(t_track, exp_y_exact, label='exact')
axs[1].legend()
axs[1].set_xlim(0,T)
#plt.title('Pauli Y')
#plt.xlabel('time')
axs[1].set_ylabel('<Y>')
#plt.show()

axs[2].plot(t_track, exp_z_RBM, label='FNN')
axs[2].plot(t_track, exp_z_exact, label='exact')
axs[2].legend()
axs[2].set_xlim(0,T)
#plt.title('Pauli Z')
#plt.xlabel('time')
axs[2].set_ylabel('<Z>')
#for ax in axs.flat:
axs[2].set(xlabel='time')
plt.show()
fig.savefig('FNNfig5.pdf')

print("--- Analysis %s seconds ---" % (time.time() - Analysis_start))
print('\n')
print("--- Total %s seconds ---" % (time.time() - start_time))
