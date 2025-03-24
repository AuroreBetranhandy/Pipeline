import glob
from operator import itemgetter
import numpy as np

all_simulations={}  #initialization

############################################### HEADER to modify
# FLASH_path="/data/home/abetranhandy/Analysis/FLASH_sims/"
FLASH_path="/data/home/abetranhandy/Analysis/Analysis_pipeline/py_files/test_file/"
GR1D_path= "/data/home/abetranhandy/Analysis/GR1D_runs/"
output_path="/scratch/abetranhandy/output/" ## In case your output file is not stored in the same place as your dat files

def plot_interp(x,y):
#     time=x
#     tau_new=savgol_filter(y,20,3)
    sorted_x,sorted_y = [list(z) for z in zip(*sorted(zip(x, y),key = itemgetter(0)))]
    # f = interpolate.interp1d(x, y,kind='nearest-up')
    time=np.linspace(np.min(sorted_x),np.max(sorted_x),len(x))                                       
    tau_new= np.interp(time,sorted_x,sorted_y)
    return time, tau_new

def plot_interp_minimize(x,y,size):
#     time=x
#     tau_new=savgol_filter(y,20,3)
    sorted_x,sorted_y = [list(z) for z in zip(*sorted(zip(x, y),key = itemgetter(0)))]
    # f = interpolate.interp1d(x, y,kind='nearest-up')
    time=np.linspace(np.min(sorted_x),np.max(sorted_x),size)                                       
    tau_new= np.interp(time,sorted_x,sorted_y)
    return time,tau_new

# 
