# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

np.random.seed(seed=0)

T = 15000 #Total time in ms
dt = 0.04 #Integration time step in ms 
nt = round(T/dt) #Time steps
N =  2000 #Number of neurons 

# Izhikevich Parameters
C = 250 #capacitance
vr = -60 #resting membrane 
b = -2 #resonance parameter 
ff = 2.5 #k parameter for Izhikevich, gain on v 
vpeak = 30 # peak voltage
vreset = -65 # reset voltage 
vt = vr + 40 - (b/ff) #threshold  
u = np.zeros(N) #initialize adaptation 
a = 0.01 #adaptation reciprocal time constant 
d = 200 #adaptation jump current 
tr = 2 #synaptic rise time 
td = 20 #decay time 
p = 0.1 #sparsity 
G = 5e3 #Gain on the static matrix with 1/sqrt(N) scaling weights.  Note that the units of this have to be in pA. 
Q = 5e3 #Gain on the rank-k perturbation modified by RLS.  Note that the units of this have to be in pA 
Irh = 0.25*ff*(vt-vr)**2

# Storage variables for synapse integration  
IPSC = np.zeros(N) #post synaptic current 
h = np.zeros(N)
r = np.zeros((N,1))
hr = np.zeros(N)
JD = np.zeros(N)

#################
#Initialization##
#################
v = vr + (vpeak-vr)*np.random.rand(N) #initial distribution 
v_ = v #These are just used for Euler integration, previous time step storage

## Target signal  COMMENT OUT TEACHER YOU DONT WANT, COMMENT IN TEACHER YOU WANT. 
zx = np.sin(2*math.pi*np.arange(nt)*dt*5*1e-3) # Target signal

k = 1 #used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.  

OMEGA = G*(np.random.randn(N,N))*(np.random.rand(N,N)<p)/(math.sqrt(N)*p) #The initial weight matrix with fixed random weights  
z = np.zeros(k) #initial approximant
BPhi = np.zeros(N) #initial decoder.  Best to keep it at 0.  
tspike = np.zeros((5*nt,2)) #If you want to store spike times, 
ns = 0 #count toal number of spikes
BIAS = 1000 #Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons 
E = (2*np.random.rand(N)-1)*Q #Weight matrix is OMEGA0 + E*BPhi'

Pinv = np.eye(N)*2 #initial correlation matrix, coefficient is the regularization constant as well 
step = 20 #optimize with RLS only every 50 steps 
imin = round(5000/dt) #time before starting RLS, gets the network to chaotic attractor 
icrit = round(10000/dt) #end simulation at this time step 
current = np.zeros(nt) #store the approximant 
RECB = np.zeros((nt,5)) #store the decoders 
REC = np.zeros((nt, 5, 2)) #Store voltage and adaptation variables for plotting 

#################
## Simulation ###
#################
for i in tqdm(range(nt)):
    # EULER INTEGRATE
    I = IPSC + E*z  + BIAS #postsynaptic current 
    v = v + dt*((ff*(v - vr)*(v - vt) - u + I)) / C  # v(t) = v(t-1)+dt*v'(t-1)
    u = u + dt*(a*(b*(v_-vr)-u)) #same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    
    index = np.where(v>=vpeak)[0] #Find the neurons that have spiked 

    # Store spike times, and get the weight matrix column sum of spikers 
    len_idx = len(index)
    if len_idx>0:
        JD = np.sum(OMEGA[:, index], axis=1) #compute the increase in current due to spiking  
        tspike[ns:ns+len_idx,:] = np.vstack((index, 0*index+dt*i)).T
        ns = ns + len_idx # total number of psikes so far

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
    z = BPhi.T @ r #approximant
    err = z - zx[i] #error 

    # RLMS 
    if i % step == 1:
        if i > imin:
            if i < icrit:
                cd = (Pinv @ r)
                BPhi = BPhi - (cd @ err.T)
                Pinv = Pinv - (cd @ cd.T) / (1.0 + r.T @ cd)

    # Store
    u = u + d*(v>=vpeak) #implements set u to u+d if v>vpeak, component by component. 
    v = v + (vreset-v)*(v>=vpeak) #implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    v_ = v # sets v(t-1) = v for the next itteration of loop
    REC[i,:] = np.vstack((v[:5], u[:5])).T
    current[i] = z.T
    RECB[i,:] = BPhi[:5]

#################
#### results ####
#################
TotNumSpikes = ns 
M = tspike[tspike[:,1]>dt*icrit,:]
AverageRate = len(M)/(N*(T-dt*icrit)*1e-3)
print("\n")
print("Total number of spikes : ", TotNumSpikes)
print("Average firing rate(Hz): ", AverageRate)

step_range = 50000
plt.figure(figsize=(6, 6))
for j in range(5):
    plt.plot(np.arange(step_range)*dt, REC[:step_range, j, 0]/(50-vreset)+j)
plt.title('Pre-Learning')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index') 
plt.savefig("Iz_pre.png")
#plt.show()

plt.figure(figsize=(6, 6))
for j in range(5):
    plt.plot(np.arange(nt-step_range, nt)*dt, REC[nt-step_range:, j, 0]/(50-vreset)+j)
plt.title('Post Learning')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index') 
plt.savefig("Iz_post.png")
#plt.show()

plt.figure(figsize=(12, 6))
plt.plot(np.arange(nt)*dt, current)
plt.plot(np.arange(nt)*dt, zx)
plt.xlim(14000,15000)
plt.title('Decoded output')
plt.xlabel('Time (ms)')
plt.ylabel('current') 
plt.savefig("Iz_post_out.png")
#plt.show()

Z = np.linalg.eig(OMEGA + np.expand_dims(E,1) @ np.expand_dims(BPhi,1).T)
Z2 = np.linalg.eig(OMEGA)
plt.figure(figsize=(6, 5))
plt.title('Weight eigenvalues')
plt.scatter(Z2[0].real, Z2[0].imag, c='r', s=5, label='Pre-Learning')
plt.scatter(Z[0].real, Z[0].imag, c='k', s=5, label='Post-Learning')
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.savefig("Iz_weight_eigenvalues.png")
#plt.show()
