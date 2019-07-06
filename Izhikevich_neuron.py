# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(seed=0)

T = 1000 #Total time in ms
dt = 0.04 #Integration time step in ms 

nt = round(T/dt) #Time steps

C = 250 #capacitance
vr = -60 #resting membrane 
b = -2 #resonance parameter 
ff = 2.5 #k parameter for Izhikevich, gain on v 
vpeak = 30 # peak voltage
vreset = -65 # reset voltage 
vt = vr + 40 - (b/ff) #threshold  
a = 0.01 #adaptation reciprocal time constant 
d = 200 #adaptation jump current 

# Initialization
v = vr #Initialize neuronal voltage with random distribtuions
v_ = v #These are just used for Euler integration, previous time step storage
u = 0
v_list = []
u_list = []

# Input
BIAS = 1000 # pA
s = np.random.randn(nt)*300 + 100 # pA

# Simulation
for i in tqdm(range(nt)):
    # Update
    I = s[i] + BIAS
    v = v + dt*((ff*(v - vr)*(v - vt) - u + I)) / C  # v(t) = v(t-1)+dt*v'(t-1)
    u = u + dt*(a*(b*(v_-vr)-u)) #same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
    
    # Reset
    u = u + d*(v>=vpeak) #implements set u to u+d if v>vpeak, component by component. 
    v = v + (vreset-v)*(v>=vpeak) #implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
    v_ = v # sets v(t-1) = v for the next itteration of loop
    
    # Save
    v_list.append(v) 
    u_list.append(u) 
    
# Plot
t = np.arange(nt)*dt*1e-3
plt.figure(figsize=(6, 5))
plt.subplot(2,1,1)
plt.title('Izhikevich neuron')
plt.plot(t, np.array(v_list))
#plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)') 
 
plt.subplot(2,1,2)
plt.plot(t, np.array(u_list))
plt.xlabel('Time (s)')
plt.ylabel('u(t)') 

plt.tight_layout()
plt.savefig('Izhikevich_neuron.png')
#plt.show()


