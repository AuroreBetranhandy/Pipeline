import numpy as np

############################# HEADER to modify on the fly 
 
time_range=[-0.01,1.1]   #### used for plots
range_of_data=[300,2000] #### plt files for convection zone calculations


# Import our modules 
                  
import Shared  ############  Paths and stuffs

import Read_dat_files ##### self descripting

import Colors  ##### That's where all colors, alphas and ticks are fixed

import Dat_files_figures  ###### Make classical figures

import Conv_figures

############ Detect convection zone & save data /!\ HEAVY 
import pns_conv_N2_2D

Read_dat_files.read_flash_files()  ##### Please verify path in Shared

Read_dat_files.read_GR1D_files()   ##### Please verify path in Shared


print(Shared.all_simulations.keys())

Colors.Set_colors_and_ticks()


Dat_files_figures.Plot_classic_Hydro(time_range)
Dat_files_figures.Plot_nu_luminosity(time_range)
Dat_files_figures.Plot_nu_energy(time_range)
# Dat_files_figures.Plot_Tau(time_range)

Conv_figures.Plot_convection_zone(time_range)



######### WIP
# GW_figures
# Nu_figures.




