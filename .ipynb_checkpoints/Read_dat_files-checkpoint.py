import Shared
import os,glob
import numpy as np
neutron_mass=1.6749286e-24
amu=1.66053907e-24
clight= 29979245800



def read_flash_files(SFHo=False):
    list_file=glob.glob(Shared.FLASH_path+'s20*.dat')
    # print(list_file)
    for i in list_file:
        base=i[len(Shared.FLASH_path):]

        if "SFHo" in base: SFHo=True
        Shared.all_simulations[base]={}
        values=np.loadtxt(i,unpack=True,usecols=(0,16,11,29,36,37,38,33,34,35,10,17,13,18)) #avoid re-opening the file 100 times
        Shared.all_simulations[base]['time']=values[0]
        Shared.all_simulations[base]['central_density']=values[1]
        Shared.all_simulations[base]['shock_radius']=values[2]
        Shared.all_simulations[base]['PNS_radius']=values[3]
        Shared.all_simulations[base]['nue_energ']=values[4]
        Shared.all_simulations[base]['nua_energ']=values[5]
        Shared.all_simulations[base]['nux_energ']=values[6]
        Shared.all_simulations[base]['nue_lumi']=values[7]
        Shared.all_simulations[base]['nua_lumi']=values[8]
        Shared.all_simulations[base]['nux_lumi']=values[9]
        Shared.all_simulations[base]['gain_net_heating']=values[11]
        Shared.all_simulations[base]['gain_mass_accretion']= values[12]
        Shared.all_simulations[base]['gain_mass']=values[13]
        
        if (SFHo==True) :
            Shared.all_simulations[base]['gain_E_bind']=values[10] - (neutron_mass - amu)*clight**2/amu * Shared.all_simulations[base]['gain_mass']
        else:
            Shared.all_simulations[base]['gain_E_bind']=values[10]
            
        try:
            time,shock=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['shock_radius'])
            Shared.all_simulations[base]['t_bounce']=time[shock>0][0]
        except: 
            print(base +"No bounce found")
            del Shared.all_simulations[base]
        # ~ globals()[i[len(files_path):-4]] = np.loadtxt(i,unpack=True,usecols=(0,11,16,17,29,33,34,35,36,37,38,18,13,10,17,9,30,31,20))    
    return
        
        
def read_GR1D_files():
    
    list_file= os.listdir(Shared.GR1D_path)
    for base in list_file:
        
        
        Shared.all_simulations[base+"_GR1D"]={}
        Shared.all_simulations[base+"_GR1D"]["time"]=np.loadtxt(Shared.GR1D_path+base+"/"+"shock_radius_t.dat",unpack=True,usecols=(0))
        Shared.all_simulations[base+"_GR1D"]["shock_radius"]=np.loadtxt(Shared.GR1D_path+base+"/"+"shock_radius_t.dat",unpack=True,usecols=(1))
        Shared.all_simulations[base+"_GR1D"]["PNS_radius"]=np.loadtxt(Shared.GR1D_path+base+"/"+"r_rho1e11.dat",unpack=True,usecols=(1))
        Shared.all_simulations[base+"_GR1D"]["central_density"]=np.loadtxt(Shared.GR1D_path+base+"/"+"rho_c_t.dat",unpack=True,usecols=(1))
        values=np.loadtxt(Shared.GR1D_path+base+"/"+"M1_flux_aveenergy_lab.dat",unpack=True,usecols=(1,2,3))
        Shared.all_simulations[base+"_GR1D"]["nue_energ"]=values[0]
        Shared.all_simulations[base+"_GR1D"]["nua_energ"]=values[1]
        Shared.all_simulations[base+"_GR1D"]["nux_energ"]=values[2]
        values=np.loadtxt(Shared.GR1D_path+base+"/"+"M1_flux_lum.dat",unpack=True,usecols=(1,2,3))
        Shared.all_simulations[base+"_GR1D"]["nue_lumi"]=values[0]
        Shared.all_simulations[base+"_GR1D"]["nua_lumi"]=values[1]
        Shared.all_simulations[base+"_GR1D"]["nux_lumi"]=values[2]
        
        try :
            time,shock=Shared.plot_interp(Shared.all_simulations[base+"_GR1D"]['time'],Shared.all_simulations[base+"_GR1D"]['shock_radius'])
            Shared.all_simulations[base+"_GR1D"]['t_bounce']=time[shock>0][0]
        except: 
            print(base +"No bounce found")
            del Shared.all_simulations[base+"_GR1D"]
            
    
    return
