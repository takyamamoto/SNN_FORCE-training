# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(seed=0)

dt = 5e-5 # (sec)
T = 0.1 # (sec)
nt = round(T/dt) #Time steps

tref = 2e-3 #Refractory time constant in seconds
tm = 1e-2 #Membrane time constant 
vreset = -65 # Reset voltage(Resting membrane potential) 
vthr = -40 # threshold voltage
vpeak = 30 #

# Initialization
v = vreset #Initialize neuronal voltage with random distribtuions
tlast = 0
v_list = []

# Input
BIAS = -40 # pA
s = np.random.randn(nt)*10 + 5 # pA

# Simulation
for i in tqdm(range(nt)):
    # Update
    I = s[i] + BIAS
    dv = ((dt*i) > (tlast + tref))*(-v + I) / tm #Voltage equation with refractory period 
    v = v + dt*dv
    
    # Check firing    
    tlast = tlast + (dt*i - tlast)*(v>=vthr) #Used to set the refractory period of LIF neurons 
    v = v + (vpeak - v)*(v>=vthr)
    
    # Save
    v_list.append(v) 
    
    # Reset
    v = v + (vreset - v)*(v>=vthr) #reset with spike time interpolant implemented.
    
# Plot
t = np.arange(nt)*dt
plt.figure(figsize=(6, 3))
plt.plot(t, np.array(v_list))
plt.title('LIF neuron')
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)') 
plt.tight_layout()
plt.savefig('LIF_neuron.png')
#plt.show()


