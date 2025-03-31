import numpy as np
import os
############################# HEADER to modify on the fly 
 
time_range=[-0.01,1.1]   #### used for plots
plt_files_range=[300,2000] #### plt files for convection zone calculations
num_of_process=60 #number of processes for paralelized convection



# Import our modules 
                  
import Shared  ############  Paths and stuffs

import Read_dat_files ##### self descripting

import Colors  ##### That's where all colors, alphas and ticks are fixed

import Dat_files_figures  ###### Make classical figures

import Conv_figures

import GW_calculations

import pns_conv_N2_2D ##### Detect convection zone & save data 

Read_dat_files.read_flash_files()  ##### Please verify path in Shared

# Read_dat_files.read_GR1D_files()   ##### Please verify path in Shared


print(Shared.all_simulations.keys())

Colors.Set_colors_and_ticks()


Dat_files_figures.Plot_classic_Hydro(time_range)
Dat_files_figures.Plot_nu_luminosity(time_range)
Dat_files_figures.Plot_nu_energy(time_range)
# Dat_files_figures.Plot_Tau(time_range)


GW_calculations.GWs(time_range)

for base in Shared.all_simulations.keys():
    if "GR1D" in base:
        print(base," No convection for 1D files")
        continue
    if os.path.isfile("N2_values/allprofiles"+base[:-4]+'.npy'): 
        print("already exists "+base)
        continue
    else:
        # pns_conv_N2_2D.run_all(base[:-4],plt_files_range)  #### [:-4] is to take of the .dat  

Conv_figures.Plot_convection_zone(time_range)

######### WIP

# Nu_figures.




