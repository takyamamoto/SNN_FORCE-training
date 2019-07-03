# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

np.random.seed(seed=0)

N = 2000  #Number of neurons 
dt = 5e-5
tref = 2e-3 #Refractory time constant in seconds 不応期
tm = 1e-2 #Membrane time constant 
vreset = -65 #Voltage reset 
vpeak = -40 #Voltage peak. 
td = 2e-2
tr = 2e-3

alpha = dt*0.1 #Sets the rate of weight change, too fast is unstable, too slow is bad as well.  
Pinv = np.eye(N)*alpha #initialize the correlation weight matrix for RLMS
p = 0.1 #Set the network sparsity 


#Target Dynamics for Product of Sine Waves
T = 15 # Simulation time (s)
imin = round(5/dt) # beginning time step of RLS training 
icrit = round(10/dt) # end time step of RLS training
step = 50 # weights update time step
nt = round(T/dt) # Simulation time step
Q = 10; G = 0.04;
zx = np.sin(2*math.pi*np.arange(nt)*dt*5) # Target signal

k = 1 # number of output unit
IPSC = np.zeros(N) #post synaptic current storage variable 
h = np.zeros(N) #Storage variable for filtered firing rates
r = np.zeros((N,1)) #second storage variable for filtered rates 
hr = np.zeros(N) #Third variable for filtered rates 
JD = np.zeros(N) #storage variable required for each spike time 
tspike = np.zeros((4*nt,2)) #Storage variable for spike times 
ns = 0 #Number of spikes, counts during simulation  
z = np.zeros(k) #Initialize the approximant 
 
v = vreset + np.random.rand(N)*(30-vreset) #Initialize neuronal voltage with random distribtuions

RECB = np.zeros((nt, 10)) #Storage matrix for the synaptic weights (a subset of them) 

OMEGA = G*(np.random.randn(N,N))*(np.random.rand(N,N)<p)/(math.sqrt(N)*p) #The initial weight matrix with fixed random weights  
BPhi = np.zeros(N) #The initial matrix that will be learned by FORCE method

#Set the row average weight to be zero, explicitly.
for i in range(N):
    QS = np.where(np.abs(OMEGA[i,:])>0)[0]
    OMEGA[i,QS] = OMEGA[i,QS] - np.sum(OMEGA[i,QS], axis=0)/len(QS)

E = (2*np.random.rand(N)-1)*Q #n

# arrays to save
REC2 = np.zeros((nt,20))
REC = np.zeros((nt,10))
current = np.zeros(nt) #storage variable for output current/approximant 

tlast = np.zeros(N) #This vector is used to set  the refractory times 
BIAS = vpeak #Set the BIAS current, can help decrease/increase firing rates.  0 is fine.

#################
## Simulation ###
#################
for i in tqdm(range(nt)):
    I = IPSC + E*z + BIAS #Neuronal Current 
    dv = ((dt*i) > (tlast + tref))*(-v + I) / tm #Voltage equation with refractory period 
    v = v + dt*dv
    
    index = np.where(v>=vpeak)[0] #Find the neurons that have spiked 
    
    # Store spike times, and get the weight matrix column sum of spikers 
    len_idx = len(index)
    if len_idx>0:
        JD = np.sum(OMEGA[:, index], axis=1) #compute the increase in current due to spiking  
        tspike[ns:ns+len_idx,:] = np.vstack((index, 0*index+dt*i)).T
        ns = ns + len_idx # total number of psikes so far

    tlast = tlast + (dt*i - tlast)*(v>=vpeak) #Used to set the refractory period of LIF neurons 
 
    # Code if the rise time is 0, and if the rise time is positive 
    if tr == 0:  
        # synapse for single exponential 
        IPSC = IPSC*math.exp(-dt/td) + JD*(len_idx>0)/td
        r = r[:,0]*math.exp(-dt/td) + (v>=vpeak)/td
    else:
        # synapse for double exponential
        IPSC = IPSC*math.exp(-dt/tr) + h*dt        
        h = h*math.exp(-dt/td) + JD*(len_idx>0)/(tr*td) #Integrate the current        
        r = r[:,0]*math.exp(-dt/tr) + hr*dt 
        hr = hr*math.exp(-dt/td) + (v>=vpeak)/(tr*td)
    
    r = np.expand_dims(r,1)
    
    # Implement RLMS with the FORCE method 
    z = BPhi.T @ r #approximant 線形結合によるデコード
    err = z - zx[i] #error 

    # RLMS 
    if i % step == 1:
        if i > imin: #iminを超えると学習開始
            if i < icrit:
                cd = (Pinv @ r)
                BPhi = BPhi - (cd @ err.T)
                Pinv = Pinv - (cd @ cd.T) / (1.0 + r.T @ cd)
    
    v = v + (30 - v)*(v>=vpeak) # 閾値を超えると30mvに
    
    REC[i] = v[:10] #Record a random voltage 
    
    v = v + (vreset - v)*(v>=vpeak) #reset with spike time interpolant implemented.
    
    current[i] = z
    RECB[i,:] = BPhi[:10]  
    REC2[i,:] = r[:20,0]

#################
#### results ####
#################
TotNumSpikes = ns 
M = tspike[tspike[:,1]>dt*icrit,:]
AverageRate = len(M)/(N*(T-dt*icrit))
print("\n")
print("Total number of spikes : ", TotNumSpikes)
print("Average firing rate(Hz): ", AverageRate)

step_range = 20000
plt.figure(figsize=(6, 6))
for j in range(5):
    plt.plot(np.arange(step_range)*dt, REC[:step_range, j]/(50-vreset)+j)
plt.title('Pre-Learning')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index') 
plt.show()

plt.figure(figsize=(6, 6))
for j in range(5):
    plt.plot(np.arange(nt-step_range, nt)*dt, REC[nt-step_range:, j]/(50-vreset)+j)
plt.title('Post Learning')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index') 
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.arange(nt)*dt, current)
plt.plot(np.arange(nt)*dt, zx)
plt.xlim(10,15)
plt.title('Decoded output')
plt.xlabel('Time (s)')
plt.ylabel('current') 
plt.show()

Z = np.linalg.eig(OMEGA + np.expand_dims(E,1) @ np.expand_dims(BPhi,1).T)
Z2 = np.linalg.eig(OMEGA)
plt.figure(figsize=(6, 5))
plt.title('Weight eigenvalues')
plt.scatter(Z2[0].real, Z2[0].imag, c='r', s=5, label='Pre-Learning')
plt.scatter(Z[0].real, Z[0].imag, c='k', s=5, label='Post-Learning')
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()
