# %% [markdown]
# # FDTD 1D simulation
# 
# Good references
# 
# * https://my.ece.utah.edu/~ece6340/LECTURES/lecture%2014/FDTD.pdf
# * https://uspas.fnal.gov/materials/10MIT/FDTD_Basics.pdf

# %% [markdown]
# ## Basic theory
# 
# ### Electomagnetics
# 
# Maxwell equation
# 
# $$\frac{\partial \mathbf{E}}{\partial t} = \frac{1}{\epsilon_0} \nabla \times \mathbf{H}\\
# \frac{\partial \mathbf{H}}{\partial t} = - \frac{1}{\mu_0} \nabla \times \mathbf{E}$$
# 
# In 1D case, 
# 
# $$\frac{\partial E_x}{\partial t} = - \frac{1}{\epsilon_0} \frac{\partial H_y}{\partial z}\\
# \frac{\partial H_y}{\partial t} = - \frac{1}{\mu_0} \frac{\partial E_x}{\partial z}$$

# %% [markdown]
# ### FDTD
# 
# FDTD is a abbreviation of Finite Difference Time Domain method.
# 
# Yee's scheme, 
# $\Delta x = L/(N-1)$
# $$P_{E} = \{x_i | x_i = i*\Delta x\}_{i=1}^N\\
# P_{H} = \{x_j | x_j = (0.5+j)*\Delta x \}_{j=1}^{N}$$

# %% [markdown]
# The gap between amplitutes of electric and magnetic fields is huge so let's introduce a normalization as 
# $$\bar{E} = \sqrt{\frac{\epsilon_0}{\mu_0}} E$$ 
# 
# Serveral conditions 
# 
# * At least, there exist 10 cell per wavelength.
# * **Courant condition**, $\Delta t \leq \frac{\Delta z}{c_0 \sqrt{d}}$, where $d=1,2,3$ is a dimension. Common setting is $\Delta t = \Delta z/ (2 c_0)$

# %% [markdown]
# ### Update progress in 1 dim
# 
# Assume that the wave trave direction in simulation is $z$ axis and electric field and magnetic field vibrating algon $x, y$ axis repectively.
# 
# Note: Except the point of sources in calculation.
# 
# The difference form of Maxwell equation in 1 dim form is 
# 
# 
# 1. $t= t_i + \frac{1}{2} \Delta t$:  Calculate $\mathbf{H}$ field using $\mathbf{E}$ value of $t=t_i$.
# 2. $t = t_i + \Delta t$: Calculate $\mathbf{E}$ field using $\mathbf{H}$ of previous step. 
# 

# %% [markdown]
# ### Boundary condition
# 
# If we just applying difference equation for updating system, the boundary just act like fixed end and the wave will be reflected at the end. To avoid such perfect reflection, we need some boundary conditions by the cases of next three.
# 
# * Absorption
# * Radiation
# * Reflection
# 
# These 3 case phenomenons are considerable. 
# 

# %% [markdown]
# 
# 
# ### Source update
# 
# * Pulse: Put constant value at points.
# * Vibration: Add independent value to update progress to source points.

# %% [markdown]
# ## Python implementation

# %%
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation as animation

from decimal import Decimal


# Optical constant in Vaccum
# source: NIST 2018 CODATA fundamental physical constant
# https://physics.nist.gov/cuu/Constants/
c0 = 299792458 # speed of light
mu0 = 1.25663706212E-6 # magnetic permeability
ep0 = 8.8541878128E-12 # electric permittivity
cm = 1E-2

# %%
#%matplotlib ipympl

# %%

# Vibrating source
w = 2*np.pi*2E-3
E0 = 5

source_index =140
# x1 x2 x3 ... xn <- L
# 0  1  2 3 <-index dx*  

wavelength_vaccum = (c0/w)*(2*np.pi)

#material setting
ep_k_coef = 1.5
mu_v_coef = 1.5
c_ratio_max = 1/np.sqrt(ep_k_coef*mu_v_coef)

min_wavelength = wavelength_vaccum*c_ratio_max 

# 1 dimension setting
nx =200
dx = min_wavelength/15
#dx = 0.02
L = dx*nx # 1 dim length
#np.linspace(0, L, nx)
p_e = np.linspace(0, L, nx)
p_m = np.linspace(0.5*dx, L+0.5*dx, nx)

# 5 
# 0 1 2 3 4
# 0 2 4 : Magnetic field
# 1, 3: Electric field 

# Material setting
ep_v = ep0*np.ones(nx*2)
mu_v = mu0*np.ones(nx*2)

ep_v[nx:int(1.5*nx)] *= ep_k_coef 
mu_v[nx:int(1.5*nx)] *= mu_v_coef 
# ---------| --- |--------

c_v = 1/np.sqrt(ep_v*mu_v)

c_e = c_v[::2] # Light speed
c_m = c_v[1::2] # initial index : final index: Step 

mat_xi = p_e[int(nx/2)] #
mat_xf = p_e[int(0.75*nx)]

# %%
# time setting
#dt = 1E5*dx/(2*c0)
dt = dx/(2*c0) # dx/(\sqrt{d} c0)
t_steps = 800 

cc_e = c_e * dt/dx
cc_m = c_m * dt/dx

# %%
# Vibrating source
#w = 2*np.pi*5E-2
#E0 = 5
#source_index =0

# %%
# Calculate FDTD 
def cal_next_fdtd(E, H, cc, bc=["ab", "ab"], s_index=None, s_func=None, args=[]):
    cc_e, cc_m = cc

    E0, E1 = E[:2]
    Em0, Em1 = E[-2:] 
    
    E[1:-1] = E[1:-1] - cc_m[1:-1] * np.diff(H)[:-1]
    
    E[0] = E1 + ((cc_m[0]-1)/(cc_m[0]+1))*(E[1]-E0) # Boundary setting for open space
    E[-1] = Em0 + ((cc_m[-1]-1)/(cc_m[-1]+1))*(E[-2] - Em1) # Boundary setting for open space

    # BC
    if s_index ==0:
        E[-1] = Em0 + ((cc_m[-1]-1)/(cc_m[-1]+1))*(E[-2] - Em1)
    elif s_index == len(E)-1:
        E[0] = E1 + ((cc_m[0]-1)/(cc_m[0]+1))*(E[1]-E0)
    else:
        #E[0] = E1 + ((cc_m[0]-1)/(cc_m[0]+1))*(E[1]-E0)
        #E[-1] = Em0 + ((cc_m[-1]-1)/(cc_m[-1]+1))*(E[-2] - Em1)
        
        E[0] = E[-1] = 0
    
    if s_index is not None:
        E[s_index] = s_func(*args)

    H[:-1] = H[:-1] - cc_e[1:] * np.diff(E) # Update magnetic field update

    return E, H
    

# %%

alpha_v=  0.8

def rect_prism(
        x_range, y_range, z_range, 
        ax3d, color_s="r", alpha=0.8):
    # TODO: refactor this to use an iterator
    xx, yy = np.meshgrid(x_range, y_range)
    #ax3d.plot_wireframe(xx, yy, z_range[0],     color=color_s)
    #ax3d.plot_surface(xx, yy, z_range[0]*np.ones(shape=xx.shape))
    #ax3d.plot_wireframe(xx, yy, z_range[1],     color=color_s)
    #ax3d.plot_surface(xx, yy, z_range[1]**np.ones(shape=xx.shape),       
    #                  color=color_s, alpha=alpha)


    yy, zz = np.meshgrid(y_range, z_range)
    #ax3d.plot_wireframe(x_range[0], yy, zz,     color="r")
    ax3d.plot_surface(x_range[0]*np.ones(shape=yy.shape), yy, zz, color="r")      
    #                    color=color_s, alpha=alpha)
    #ax3d.plot_wireframe(x_range[1], yy, zz,     color="r")
    ax3d.plot_surface(x_range[1]*np.ones(shape=yy.shape), yy, zz, color="r")       
    #                    color=color_s, alpha=alpha)

    #xx, zz = np.meshgrid(x_range, z_range)
    #ax3d.plot_wireframe(xx, y_range[0], zz,     color="r")
    #ax3d.plot_surface(xx, y_range[0]*np.ones(shape=zz.shape), zz,       
    #                    color=color_s, alpha=alpha)
    #ax3d.plot_wireframe(xx, y_range[1], zz,     color="r")
    #ax3d.plot_surface(xx, y_range[1]*np.ones(shape=zz.shape), zz,       
    #                    color=color_s, alpha=alpha)


def update(frame, ax3d, Ey, Bz, plot_surf):
    #f_i = int((dtl*frame)%length)
    ##f_i = frame
    Ey, Bz = cal_next_fdtd(Ey, Bz, [cc_e, cc_m], s_index=source_index, s_func=lambda x: E0*np.sin(w*x), args=[frame])
    
    ax3d.cla()
    #plot_surf.remove()
    alpha_v = 0.5
    plot_surf = ax3d.plot_surface(
        mat_xi*np.array([[1,1],[1,1]]), 
        np.array([[E0, -E0],[E0, -E0]]), 
        np.array([[E0, E0],[-E0, -E0]]),
        color="r",
        #cmap="afmhot_r", 
        alpha=alpha_v) 
    plot_surf = ax3d.plot_surface(
        mat_xf*np.array([[1,1],[1,1]]), 
        np.array([[E0, -E0],[E0, -E0]]), 
        np.array([[E0, E0],[-E0, -E0]]),
        color="r",
        #cmap="afmhot_r", 
        alpha=alpha_v) 
    
    ax3d.plot(p_e, axis_point_1, Ey)
    ax3d.plot(p_m, Bz, axis_point_2)

    ax3d.set_zlim([-E0, E0])
    ax3d.set_ylim([-E0, E0])
    ax3d.set_title(f"t:{Decimal(frame*dt):1.4E}")
    #ax3d.legend(["$E_z$", "B_y", "matter_face_1", "matter_face_2"])
    #print(frame, max_f, end="\r")
    #Ey, Bz =(np.zeros(Ey.size) , np.zeros(Ey.size) )if not (frame % max_f) else Ey, Bz
    return E_plot, B_plot, plot_surf

# %%
Ey = np.zeros(p_e.size)
Bz = np.zeros(p_m.size)
# x, y, z
axis_point_1 = np.zeros(Ey.size)
axis_point_2 = np.zeros(Bz.size)

# %%
# Matplotlib animating

box_aspect_set = {"aspect":(10, 3, 3),
                  "zoom" : 0.8}

fig2 = plt.figure()
ax3d = fig2.add_subplot(1,1, 1, projection="3d")
ax3d.legend(["$E$", "$B$"])
plot_surf = ax3d.plot_surface(
        mat_xi*np.array([[1,1],[1,1]]), 
        np.array([[E0, -E0],[E0, -E0]]), 
        np.array([[E0, E0],[-E0, -E0]]), alpha=0.8) 
E_plot = ax3d.plot(p_e, axis_point_1, Ey)
B_plot = ax3d.plot(p_m, Bz, axis_point_2)

ax3d.set_title("test")
ax3d.set_box_aspect(**box_aspect_set )
ani = animation.FuncAnimation(
    fig2, 
    func=update, 
    fargs=[ax3d, Ey, Bz, plot_surf],
    frames=2*t_steps, interval=10,
    )
plt.show()


