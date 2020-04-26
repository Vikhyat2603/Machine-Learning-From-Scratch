import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Generate 100 random values from 1 to 100
y = np.random.randint(low=1, high=100, size=100) 

newY1 = []
newY2 = []
newY3 = []

beta = 0.9

prev = np.zeros(3) # Three separate methods of correction
prev[-1] = y[0] # Initialise prev with the first element of y

for i in range(len(y)):    
    t = beta*prev + (1-beta)*y[i]
    
    newY1.append(t[0])
    newY2.append(t[1]/(1-beta**(i+1))) # Bias correction
    newY3.append(t[2])
    
    prev = t

plt.close('all')

fig = plt.figure(figsize = (14,4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
    
ax1.plot(y)
ax1.plot(newY1)
ax1.title.set_text('No correction')

ax2.plot(y)
ax2.plot(newY2)
ax2.title.set_text('Bias correction')

ax3.plot(y)
ax3.plot(newY3)
ax3.title.set_text('Intialise v0 as y[0]')

plt.tight_layout()
plt.show()