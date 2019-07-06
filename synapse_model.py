# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

dt = 5e-5 # (sec)
td = 2e-2 #synaptic decay time
tr = 2e-3 #synaptic rise time
T = 0.1 # (sec)
nt = round(T/dt) #Time steps

# synapse for single exponential 
r = 0 # initial
single_r = [] 
for i in range(nt):    
    if i == 0:
        spike = 1
    else:
        spike = 0
    single_r.append(r)
    r = r*math.exp(-dt/td) + spike/td
    #r = r*(1-dt/td) + spike/td

# synapse for double exponential
r = 0; hr = 0; # initial
double_r = []
for i in range(nt):    
    if i == 0:
        spike = 1
    else:
        spike = 0
    double_r.append(r)
    r = r*math.exp(-dt/tr) + hr*dt 
    hr = hr*math.exp(-dt/td) + spike/(tr*td)
    #r = r*(1-dt/tr) + hr*dt 
    #hr = hr*(1-dt/td) + spike/(tr*td)

# Plot
t = np.arange(nt)*dt
plt.figure(figsize=(5, 4))
plt.plot(t, np.array(single_r), label="single exponential")
plt.plot(t, np.array(double_r), label="double exponential")
plt.title('Synapse models')
plt.xlabel('Time (s)')
plt.ylabel('Post-synaptic current (pA)') 
plt.legend()
plt.tight_layout()
plt.savefig('Synapse.png')
#plt.show()

