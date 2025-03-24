
import yt 
import Shared
from yt import derived_field
import numpy as np
from scipy.interpolate import interp1d
from os.path import exists
from scipy import signal, linalg
from yt.funcs import mylog
mylog.setLevel(50)

import h5py

import astropy.constants as cn
import abel_modified
import glob

from scipy.interpolate import RegularGridInterpolator as rgi

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


def polarize(filename):
    radius_carth,theta,rho,x2,y2,press,mass,vrad,gamc,ye,temp,time,gpot,c_s,shock,rnue,rnua,rnux=open_data(filename)
    
    
    
    
    
    # r2,theta2=cart2polar(x, y)theta,r
    
    x=x2*(x2.shape[1]/x2.max())
    y=y2*((y2.shape[0]/2)/(y2.max()))


    
    dens,r2,theta=abel_modified.reproject_image_into_polar(rho,x,y)
    pressure,r2,theta=abel_modified.reproject_image_into_polar(press,x,y)
    r_shock,r2,theta=abel_modified.reproject_image_into_polar(shock,x,y)
    vel_rad,r2,theta=abel_modified.reproject_image_into_polar(vrad,x,y)
    gamc,r2,theta=abel_modified.reproject_image_into_polar(gamc,x,y)
    cell_mass,r2,theta=abel_modified.reproject_image_into_polar(mass,x,y)
    ye,r2,theta=abel_modified.reproject_image_into_polar(ye,x,y)
    temp,r2,theta=abel_modified.reproject_image_into_polar(temp,x,y)
    grav_pot,r2,theta=abel_modified.reproject_image_into_polar(gpot,x,y)
    sound_speed,r2,theta=abel_modified.reproject_image_into_polar(c_s,x,y)
    r_nue,r2,theta=abel_modified.reproject_image_into_polar(rnue,x,y)
    r_nua,r2,theta=abel_modified.reproject_image_into_polar(rnua,x,y)
    r_nux,r2,theta=abel_modified.reproject_image_into_polar(rnux,x,y)
    
    r=r2*np.sqrt(x2.max()**2+(y2.max())**2)/np.sqrt(x.max()**2+y.max()**2)
    # r=r2*np.sqrt(x2.max()**2+(y2.max())**2)/len(x2)*len(r2)/1e5
    
    
    
    dye=np.gradient(ye,axis=0)
    drho=np.gradient(np.log(dens),axis=0)
    dtemp=np.gradient(np.log(temp),axis=0)
    dp=np.gradient(np.log(pressure),axis=0)
    dr=np.gradient(r,axis=0)
    dtheta=np.gradient(theta,axis=1)
    dgpot=np.gradient(grav_pot,axis=0)
    
    vol=2*np.pi*r*dr*(1/len(theta[0,:])*r)
    mass=dens*vol
    #     print(mass,vol,dens)
    #N2=dgpot/dr*(1/gamc*dp/dr - drho/dr )#+ drho**2/dtemp/dr)
    N2=dgpot/dr*(1/(sound_speed**2*dens/pressure)*dp/dr - drho/dr )
    kinetic= mass*vel_rad**2
    r=r/1e5
    diff_vel=(vel_rad-np.mean(vel_rad,axis=0))

    mean_N2=np.nanmean(N2,axis=1)
    mean_rnu=np.nanmean(r_nue+r_nua,axis=1)
    # mean_rnua=np.nanmean(r_nua,axis=1)
    
    
    mask_minus=np.logical_and(mean_N2>0,r[:,0]<25)
  
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
    tot_mass[r<min_val]=np.nan
    tot_mass[N2>0]=np.nan
    
    mass_shock=mass.copy()
    mass_shock[r<shock_conv_min]=np.nan
    mass_shock[r>shock_conv_max]=np.nan
    
    return np.nansum(kinetic_PNS),max_val,min_val,np.nansum(tot_mass),time,np.nansum(kin_shock), np.nansum(mass_shock),shock_conv_min,shock_conv_max


def blip_bloup(i):
    filename = Shared.output_path+base+'_hdf5_plt_cnt_'+str(i).zfill(4)        # ~ filename = base+'_hdf5_plt_cnt_'+str(i).zfill(4)

        
    if not os.path.isfile(filename):
        return np.zeros(9)
    list_of_dict=polarize(filename)
    return list_of_dict
    
def run_all(base,allprofiles):
    print(base)
    allprofiles[base] = {}
    
    filename = Shared.output_path+base+'_hdf5_plt_cnt_0300'        # ~ filename = base+'_hdf5_plt_cnt_'+str(i).zfill(4)
    if not "cylindrical" in str(h5py.File(filename)['string scalars'][:][-1][-2]):
        print( "1D file")
        return
    time_range=np.arange(nstarts[base],nends[base])
    # print(time_range)
    pool=Pool(60)
    procs=list()
    M= np.array(pool.map(blip_bloup,time_range))
    pool.close()

    # print(M,"run_all")
    sorted_M= M[M[:,4].argsort()]
    allprofiles[base]["kin_erg"]= sorted_M[:,0]
    allprofiles[base]["conv_max_rad"]=sorted_M[:,1]
    allprofiles[base]["conv_min_rad"]=sorted_M[:,2]
    allprofiles[base]["total_mass"]=sorted_M[:,3]
    allprofiles[base]['time']= sorted_M[:,4]
    allprofiles[base]["kinetic_shock"]=sorted_M[:,5]
    allprofiles[base]["mass_shock"]=sorted_M[:,6]
    allprofiles[base]["r_shock_in"]=sorted_M[:,7]
    allprofiles[base]["r_shock_out"]=sorted_M[:,8]
    print(allprofiles[base])
    np.save("N2_values/allprofiles"+base, allprofiles[base])





file_list=glob.glob(Shared.FLASH_path+'*.dat')


allprofiles={}
bases=[]
nstarts={}
nends={}

for i in file_list:

    bases.append(i[len(Shared.FLASH_path):-4])
    nends[i[len(Shared.FLASH_path):-4]]=2000
    nstarts[i[len(Shared.FLASH_path):-4]]=0
    if os.path.isfile("N2_values/allprofiles"+bases[-1]+'.npy'):
        print("already exists "+bases[-1])
        bases.remove(bases[-1])
# ~ run_all("s20_simp_SFHo_Hann8")
for base in bases:
    
    try:
        run_all(base,allprofiles)
    except: 

        print(base, " failed")

