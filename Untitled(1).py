#!/usr/bin/env python
# coding: utf-8

# In[2]:


import yt 
from yt import derived_field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from os.path import exists
from scipy import signal, linalg
from matplotlib.colors import LogNorm, SymLogNorm
from yt.funcs import mylog
mylog.setLevel(50)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
import h5py

import astropy.constants as cn
import abel_modified
import glob

from scipy.interpolate import RegularGridInterpolator as rgi
import matplotlib.font_manager as font_manager
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import os,gc
from multiprocessing import Pool,Manager,Process
import subprocess

from scipy import interpolate
from functools import partial
G=cn.G.cgs.value

from scipy.signal import savgol_filter
import warnings

warnings.simplefilter('ignore')


@derived_field(name='mycostheta',sampling_type='cell')
def _mycostheta(field,data):
    return data['z']/np.sqrt(data['r']**2+data['z']**2)

@derived_field(name='mysintheta',sampling_type='cell')
def _mysintheta(field,data):
    return data['r']/np.sqrt(data['r']**2+data['z']**2)
    
    
@derived_field(name='radius',sampling_type='cell')
def _radius(field,data):
    return np.sqrt(data['r']**2+data['z']**2)

@derived_field(name='vrad',units='cm/s',sampling_type='cell')
def _vrad(field,data):
    return data['vely']*data['mycostheta']+data['velx']*data['mysintheta']

@derived_field(name='vmag',units='cm/s',sampling_type='cell')
def _vmag(field,data):
    return np.sqrt(data['vely']**2+data['velx']**2)

@derived_field(name='vtheta',units='cm/s',sampling_type='cell')
def _vtheta(field,data):
    return data['velx']*data['mycostheta']-data['vely']*data['mysintheta']

@derived_field(name='rho_p_eint',units='g/cm**3',sampling_type='cell')
def _rho_p_eint(field,data):
    return data['density']+data['density']*(data['eint'].v/29979245800.0**2 - 2.86969e+19/29979245800.**2)





from abel_modified import reproject_image_into_polar
# from abel_modified import reproject_image_into_polar

# from pns_conv_N2 import polarize
# from pns_conv_N2 import open_data



# In[3]:


# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import map_coordinates


def reproject_image_into_polar(data,x,y, origin=None, Jacobian=False,
                               dr=1, dt=None):
    """
    Reprojects a 2D numpy array (**data**) into a polar coordinate system,
    with the pole placed at **origin** and the angle measured clockwise from
    the upward direction. The resulting array has rows corresponding to the
    radial grid, and columns corresponding to the angular grid.

    Parameters
    ----------
    data : 2D np.array
        the image array
    origin : tuple or None
        (row, column) coordinates of the image origin. If ``None``, the
        geometric center of the image is used.
    Jacobian : bool
        Include `r` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    dr : float
        radial coordinate spacing for the grid interpolation.
        Tests show that there is not much point in going below 0.5.
    dt : float or None
        angular coordinate spacing (in radians).
        If ``None``, the number of angular grid points will be set to the
        largest dimension (the height or the width) of the image.

    Returns
    -------
    output : 2D np.array
        the polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of angular coordinates

    Notes
    -----
    Adapted from:
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system

    """
    ny, nx = y.shape
    if origin is None:
        origin = (ny // 2, 0)
   

    # Determine what the min and max r and theta coords will be...
    # ~ x, y = index_coords(data, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical
    dr = np.gradient(r,axis=1).min()
    # ~ nr = int(np.ceil((r.max() - r.min()) / dr))
    nr= min(ny,nx)
    nt=2*nr
    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in radians
        nt = int(np.ceil((theta.max() - theta.min()) / dt))

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=True)


    
    print(r.min(),r.max(),x.min(),y.min(),np.sqrt(x.min()**2+y.min()**2),np.sqrt(x.max()**2+y.max()**2),r_i.max(),r_i.min())
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)





    
    # ~ print(theta_grid.shape,r_grid.shape)
    # Convert the r and theta grids to Cartesian coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    # then to a 2×n array of row and column indices for np.map_coordinates()
    rowi = (origin[0] - Y).flatten()
    coli = (X + origin[1]).flatten()
    coords = np.vstack((rowi, coli))

    # Remap with interpolation
    # (making an array of floats even if the data has an integer type)
    zi = map_coordinates(data, coords, output=float)
    output = zi.reshape((nr, nt))
    # ~ print(output.shape)
    if Jacobian:
        output *= r_i[:, np.newaxis]

    return output, r_grid, theta_grid



def index_coords(data, origin=None):
    """
    Creates `x` and `y` coordinates for the indices in a numpy array, relative
    to the **origin**, with the `x` axis going to the right, and the `y` axis
    going `up`.

    Parameters
    ----------
    data : numpy array
        2D data. Only the array shape is used.
    origin : tuple or None
        (row, column). Defaults to the geometric center of the image.

    Returns
    -------
        x, y : 2D numpy arrays
    """
    ny, nx = y.shape
    if origin is None:
        origin_x, origin_y = 0, ny // 2
    else:
        origin_y, origin_x = origin
        # wrap negative coordinates
        if origin_y < 0:
            origin_y += ny
        if origin_x < 0:
            origin_x += nx

    x, y = np.meshgrid(np.linspace(0,float(nx),nx) - origin_x,
                       origin_y - np.linspace(0,float(ny),ny))
    print(x.min(),x.max(),y.min(),y.max())
    return x, y



def cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar.

    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates

    Returns
    -------
    r, theta : floats or arrays
        Polar coordinates

    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta



def polar2cart(r, theta):
    """
    Transform polar coordinates to Cartesian.

    Parameters
    -------
    r, theta : floats or arrays
        Polar coordinates

    Returns
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    """
    y = r * np.cos(theta)   # θ referenced to vertical
    x = r * np.sin(theta)
    return x, y


# In[4]:


def flip_0(data):
    data[np.isnan(data)]=0
    data= data+np.flip(data,axis=1)
    return data



def open_data(filename):
    
    test=h5py.File(filename)
    time=test['real scalars'][2][-1]
    r_shock=test['real scalars'][2][-1]
    nblockx=test['integer runtime parameters'][74][-1]
    refine=test['integer runtime parameters'][62][-1]
    nblocky=nblockx*2
    ds_2d = yt.load(filename)
    data = ds_2d.slice(2,0)
    npoints=3000
    rmax=1e9/nblocky/nblockx/2**refine*npoints/1e5
    #time=float(np.nanmean(data[('gas', 'dynamical_time')]))

    #nsize=int((ds_2d.all_data().icoords.shape)[0]**(1./3.))*3
    nsize=npoints
    data = data.to_frb((rmax,"km"),nsize,center=(0,0),periodic=True)
    x = data[('index','r')].d[:,int(nsize/2)+1:]
    y = data[('index','z')].d[:,int(nsize/2)+1:]
    
    radius = data[('radius')].d[:-1,int(nsize/2)+1:-1] 
    press = data['pres'].d[:-1,int(nsize/2)+1:-1]
    mass = data['cell_mass'].d[:-1,int(nsize/2)+1:-1]
    rho = data['dens'].d[:-1,int(nsize/2)+1:-1]
    gamc = data['gamc'].d[:-1,int(nsize/2)+1:-1]/29979245800
    vrad = data['vrad'].d[:-1,int(nsize/2)+1:-1]
    ye= data['ye  '].d[:-1,int(nsize/2)+1:-1] 
    gpot = data['gpot'].d[:-1,int(nsize/2)+1:-1]
    temp= data['temp'].d[:-1,int(nsize/2)+1:-1] 
    c_s = data['sound_speed'].d[:-1,int(nsize/2)+1:-1]
    shock = data['shok'].d[:-1,int(nsize/2)+1:-1]
    rnua = data['fnua'].d[:-1,int(nsize/2)+1:-1]
    rnue = data['fnue'].d[:-1,int(nsize/2)+1:-1]
    rnux = data['fnux'].d[:-1,int(nsize/2)+1:-1]
    theta=np.arccos(data['mycostheta'].d[:-1,int(nsize/2)+1:-1])
    del ds_2d, data

    return radius,theta,rho,x,y,press,mass,vrad,gamc,ye,temp,time,gpot,c_s,shock,rnue,rnua,rnux


# In[33]:


base="s20_ref_SRO3"
i=375

# for i in np.arange(350,1200,5):
#     filename = "/scratch/abetranhandy/output/"+base+'_hdf5_plt_cnt_'+str(i).zfill(4)
filename = "/scratch/abetranhandy/output/"+base+'_hdf5_plt_cnt_'+str(i).zfill(4)

# polarize(filename)


# In[34]:


radius_carth,theta,rho,x2,y2,press,mass,vrad,gamc,ye,temp,time,gpot,c_s,shock,rnue,rnua,rnux=open_data(filename)





# r2,theta2=cart2polar(x, y)theta,r

x=x2*(x2.shape[1]/x2.max())
y=y2*((y2.shape[0]/2)/(y2.max()))


dens,r2,theta=reproject_image_into_polar(rho,x,y)
pressure,r2,theta=reproject_image_into_polar(press,x,y)
r_shock,r2,theta=reproject_image_into_polar(shock,x,y)
vel_rad,r2,theta=reproject_image_into_polar(vrad,x,y)
gamc,r2,theta=reproject_image_into_polar(gamc,x,y)
cell_mass,r2,theta=reproject_image_into_polar(mass,x,y)
ye,r2,theta=reproject_image_into_polar(ye,x,y)
temp,r2,theta=reproject_image_into_polar(temp,x,y)
grav_pot,r2,theta=reproject_image_into_polar(gpot,x,y)
sound_speed,r2,theta=reproject_image_into_polar(c_s,x,y)
r_nue,r2,theta=reproject_image_into_polar(rnue,x,y)
r_nua,r2,theta=reproject_image_into_polar(rnua,x,y)
r_nux,r2,theta=reproject_image_into_polar(rnux,x,y)

r=r2*np.sqrt(x2.max()**2+(y2.max())**2)/np.sqrt(x.max()**2+y.max()**2)
# r=r2*np.sqrt(x2.max()**2+(y2.max())**2)/len(x2)*len(r2)/1e5



dye=np.gradient(ye,axis=0)
drho=np.gradient(np.log(dens),axis=0)
dtemp=np.gradient(np.log(temp),axis=0)
dp=np.gradient(np.log(pressure),axis=0)
dr=np.gradient(r,axis=0)
dtheta=np.gradient(theta,axis=1)
dgpot=np.gradient(grav_pot,axis=0)

vol=2*np.pi*r*dr*dtheta  # km
mass=dens*vol
#     print(mass,vol,dens)
#N2=dgpot/dr*(1/gamc*dp/dr - drho/dr )#+ drho**2/dtemp/dr)
N2=dgpot/dr*(1/(sound_speed**2*dens/pressure)*dp/dr - drho/dr )
kinetic= mass*vel_rad**2
r=r/1e5
diff_vel=(vel_rad-np.mean(vel_rad,axis=0))
#     f = interpolate.interp1d(r[:,0], np.mean(N2,axis=1))
#     time=np.linspace(r[:,0].min(),r[:,0].max(),400)
#     mean_N2= f(time)
mean_N2=np.nanmean(N2,axis=1)
mean_rnu=np.nanmean(r_nue+r_nua,axis=1)
# mean_rnua=np.nanmean(r_nua,axis=1)


mask_minus=np.logical_and(mean_N2>0,r[:,0]<25)

# mask_minus=np.logical_and(mean_N2>0,r[:,0]<15)
# ~ print(shock_rm.shape)
if len(r[mask_minus])>0:
    min_val= r[mask_minus].max()
else: 
    min_val=0
    

    
mask_plus=np.logical_and(mean_N2>0,r[:,0]>min_val) 

if len(r[mask_plus])>0:
    max_val=r[mask_plus].min()
else:
    max_val=min_val


r_shock[r_shock<0.8]=np.nan
mean_shock=np.nanmean(r_shock,axis=1)
shock_r=np.nanmean(r[:,0][mean_shock>0])
shock_rm=np.full((len(theta[0,:])),shock_r)
mask_gain=np.logical_and(mean_N2<0,np.logical_and(r[:,0]>max_val,mean_rnu>0))  


shock_conv_max= np.mean(shock_rm)
if len(r[:,0][mask_gain])>0:
    shock_conv_min=r[:,0][mask_gain].min()
else: 
    shock_conv_min=max_val
    
kinetic_PNS=kinetic.copy()
# kinetic_PNS[N2>0]=np.nan
kinetic_PNS[r>max_val]=np.nan
kinetic_PNS[r<min_val]=np.nan



kin_shock=kinetic.copy()
# kin_shock[N2>0]=np.nan
kin_shock[r<shock_conv_min]=np.nan
kin_shock[r>shock_conv_max]=np.nan


tot_mass=mass.copy()
# tot_mass[r>max_val]=np.nan
tot_mass[r<min_val]=np.nan
tot_mass[N2>0]=np.nan

mass_shock=mass.copy()
# mass_shock[N2>0]=np.nan
mass_shock[r<shock_conv_min]=np.nan
mass_shock[r>shock_conv_max]=np.nan


# In[7]:


print(x2.shape,r2.T.shape)
r.min(),r.max(),x.min(),y.min(),np.sqrt(x.min()**2+y.min()**2),np.sqrt(x.max()**2+y.max()**2)


# In[36]:


# fig,ax=plt.subplots(1,1,figsize=(10,10))
# cs=ax.imshow(vrad,norm=SymLogNorm(linthresh=np.nanmax(abs(vrad))*1e-2),cmap='seismic',extent=(x.min(),x.max(),y.min(),y.max()))  
# fig.colorbar(cs, ax=ax, shrink=0.6)
# r=r/1e5
fig,ax=plt.subplots(2,3,figsize=(30,20),subplot_kw={'projection': 'polar'})

# cs=ax[0].pcolormesh(theta,r,abs(diff_vel),
#             norm=SymLogNorm(linthresh=np.nanmax(abs(diff_vel))/1e2,vmin=-np.nanmax(abs(diff_vel)),vmax=np.nanmax(abs(diff_vel)))
#             # ~ norm=LogNorm()
#             ,cmap='seismic')    
# cs=ax[0].pcolormesh(theta,r,diff_vel,
# 			# norm=SymLogNorm(linthresh=np.nanmax(abs(diff_vel))/1e7,vmin=-np.nanmax(abs(diff_vel)),vmax=np.nanmax(abs(diff_vel)))
# 			norm=LogNorm()
# 			,cmap='seismic')
cs=ax[0,0].pcolormesh(theta,r,dens,
            # norm=SymLogNorm(linthresh=1e12)
            # norm=LogNorm(vmin=1e11,vmax=1e15)
             norm=LogNorm()
            ,cmap='seismic')
im=ax[0,1].pcolormesh(theta,r,r_nua,
            # norm=SymLogNorm(linthresh=1e12)
            norm=LogNorm(vmin=1e-19,vmax=1e-11)
            ,cmap='Greens')
im2=ax[0,2].pcolormesh(theta,r,r_nux/4,
            # norm=SymLogNorm(linthresh=1e12)
            norm=LogNorm(vmin=1e-19,vmax=1e-11)
            ,cmap='Reds')
# im=ax[1].pcolormesh(theta[:,:],r[:,:],-N2[:,:],norm=SymLogNorm(linthresh=np.nanmax(abs(N2))/1e11,\
# 			vmin=-np.nanmax(abs(N2)),vmax=np.nanmax(abs(N2))),cmap='seismic')


vs=ax[1,0].pcolormesh(theta,r,vel_rad,norm=SymLogNorm(linthresh=np.nanmax(abs(vel_rad))*1e-2),cmap='seismic')    

im3=ax[1,1].pcolormesh(theta,r,N2
            ,norm=SymLogNorm(linthresh=np.nanmax(abs(N2))*1e-2)
            # ,norm=LogNorm()
            , shading='nearest'
            ,cmap='seismic')


ax[1,1].fill_between(theta[0,:],shock_conv_min,shock_conv_max,facecolor='y',alpha=0.5)
ax[1,0].fill_between(theta[0,:],shock_conv_min,shock_conv_max,facecolor='y',alpha=0.5)
ax[1,0].fill_between(theta[0,:],min_val,max_val,facecolor='g',alpha=0.5)
ax[1,1].fill_between(theta[0,:],min_val,max_val,facecolor='g',alpha=0.5)

# ax[1].fill_between(theta[0,:],min_val,max_val,facecolor='w',alpha=0.5)
ax[1,0].plot(theta[0,:],np.zeros(len(theta[0,:]))+50,'w')#(theta,r,r_shock,cmap='Reds')
ax[1,0].plot(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')
ax[1,1].plot(theta[0,:],shock_rm,'w')#(theta,r,r_shock,cmap='Reds')



ax[0,0].set_rmax(150)
ax[0,1].set_rmax(150)
ax[0,2].set_rmax(150)
ax[1,0].set_rmax(150)
ax[1,1].set_rmax(150)

ax[0,0].set_title('fnue')
ax[0,1].set_title('fnua')
ax[0,2].set_title('fnux')
ax[1,0].set_title('N2')
ax[1,1].set_title('Temp')
fig.colorbar(cs, ax=ax[0,0], shrink=0.6)
fig.colorbar(im, ax=ax[0,1], shrink=0.6)
fig.colorbar(im2,ax=ax[0,2], shrink=0.6)
fig.colorbar(vs, ax=ax[1,0], shrink=0.6)
fig.colorbar(im3,ax=ax[1,1], shrink=0.6)
ax[1,2] = fig.add_subplot(236, polar=False)
vrrp=ax[1,2].pcolormesh(x2/1e5,y2/1e5,vrad,norm=SymLogNorm(linthresh=np.nanmax(abs(vrad))*1e-2),cmap='seismic')
ax[1,2].set_xlim([0,150])
ax[1,2].set_ylim([-150,150])
print(np.sqrt(x2.max()**2+y2.max()**2)/1e5,x.max(),y.max(),r.max(),r2.max())
fig.colorbar(vrrp,ax=ax[1,2], shrink=0.6)

# ax2=fig.add_subplot(1,2,1)    
#   # ~ mean_N2=savgol_filter(mean_N2,25,2)    
#  #     print(np.nansum(tot_mass))    
# ax2.semilogy(r[:,0][r[:,0]<50],(mean_N2[r[:,0]<50]),'r')
# ax2.semilogy(r[:,0][r[:,0]>shock_conv_min],(mean_N2[r[:,0]>shock_conv_min]),'b')

# ax2.axhline(0,0,300,c='k')
# ax2.set_ylim([-1e5,0])
# ax2.set_xlim([0,50])
# plt.show()
# plt.plot(r[:,50])
# plt.savefig('figures/neutrino_and_temp_'+filename[29:]+".png")
# plt.close(fig)
# except:
    # print(filename[-4:]+" fig didn't work")
# for i in list(locals().keys()):
#      exec('del ' + i)
# gc.collect()
# return np.nansum(kinetic_PNS),max_val,min_val,np.nansum(tot_mass),time,np.nansum(kin_shock), np.nansum(mass_shock),shock_conv_min,shock_conv_max


# In[18]:


plt.plot(theta[0,1:]-theta[0,:-1])
print(len(theta[0,:]))


# In[8]:


import scipy
print(vrad.ndim)
test,r_polar2,theta=reproject_image_into_polar(radius_carth)


x2,y2=polar2cart(r2,theta)
print(x2.min())
print(r.min(), np.sqrt(x**2+y**2).min())
plt.pcolormesh(theta,r,test/1e5
            # norm=SymLogNorm(linthresh=1e12)
            # ,norm=LogNorm()
            , shading='nearest'
            ,cmap='seismic')


