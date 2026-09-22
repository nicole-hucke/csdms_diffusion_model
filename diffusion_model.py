#!/usr/bin/env python
# coding: utf-8

# # 1D Diffusion Model
# * This is an example on how we develop a one-dimensional numerical model on diffusion
# * It assumes a constant diffusivity
# * It uses a regular grid
# * It has a step function as an initial profile
# * It has a fixed boundary condition

# ### This is the diffusion equation

# $$ \frac{\partial z}{\partial t} = D\frac{\partial^2 z}{\partial x^2} $$

# Where $z$ is elevation, $t$ is time, $x$ is the spatial dimension, and $D$ is the hillslope diffusivity.

# ### This is the discretized version of the diffusion model we'll solve: 

# $$ z^{t+1}_x = z^t_x + {D \Delta t \over \Delta x^2} (z^t_{x+1} - 2z^t_x + z^t_{x-1}) $$

# This is the FTCS discretization scheme from Slingerland and Kump (2011)

# We'll use two libraries, NumPy and Matplotlib that aren't a part of the core Python distribution

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# #### Here are some hidden NumPy examples 
# You can use the arrows to display them, but I want to keep them hidden for the sake of the example

# In[ ]:


x = np.zeros(5)
print(x) 


# In[ ]:


y = np.arange(5, dtype=float) + 10
y


# In[ ]:


y[0]


# In[ ]:


y[4]


# In[ ]:


y[-1]


# In[ ]:


y[1:4]


# In[ ]:


y[1:]


# ### Start by setting two fixed model parameters, the diffusivity and the size of the model domain:

# In[ ]:


D = 100
Lx = 300


# Set up the model grid using NumPy arrays

# In[ ]:


dx = 0.5 # grid spacing
x = np.arange(start=0, stop=Lx, step=dx) # the one-dimensional grid
nx = len(x) # length of the grid 


# ### Set the initial conditions for the model
# The elevation $z$ is a step function with a high value on the left, a low value on the right, and a cliff at the center of the domain.

# In[ ]:


z = np.zeros_like(x)
z_hi = 500.0
z_lo = 0.0


# In[ ]:


z[0:int(Lx/2)] = z_hi
z[int(Lx/2):] = z_lo


# In[ ]:


z[x <= Lx/2] = z_hi
z[x > Lx/2] = z_lo


# ### Plot the initial hillslope profile

# In[ ]:


plt.figure()
plt.plot(x,z,"r")
plt.xlabel("x") 
plt.ylabel("z")
plt.title("Initial hillslope profile")
plt.show() 


# ### Set the number of time steps in the model
# Calculate a stable time step using a stabiltiy criterion.

# In[ ]:


nt = 5000
dt = 0.5 * dx**2 / D


# In[ ]:


dt # we assume that these are in seconds 


# ### Loop over the time steps of the model, solving the diffusion equation using the FTCS scheme described above. 
# We'll use an array operation on the variable $z$

# In[ ]:


for t in range(0, nt):
	z[1:-1] += D * dt / dx ** 2 * (z[:-2] - 2*z[1:-1] + z[2:])


# ### Plot the evolved hillslope profile

# In[ ]:


plt.figure()
plt.plot(x,z,"b")
plt.xlabel("x") 
plt.ylabel("z")
plt.title("Final hillslope profile")
plt.show() 

