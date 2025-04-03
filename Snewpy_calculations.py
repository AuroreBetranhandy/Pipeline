import snewpy
import numpy as np
import scipy 
import os
from snewpy import snowglobes
import warnings
warnings.simplefilter('ignore')
import matplotlib.pylab as plt
import Shared
from astropy import units as u



names=[]
numbers=[]
save_number=[]

detec=np.loadtxt(Shared.SNOwGLoBES_path+'detector_configurations.dat',unpack=True, dtype=str)
detector_name=detec[0]
detector_mass=detec[1]
detector_norma=detec[2]
detector_name=np.delete(detector_name,np.argwhere(detector_name == 'halo1'))
detector_name=np.delete(detector_name,np.argwhere(detector_name == 'halo2'))



def neutrino_all_sim_all_detectors():
    for base in Shared.all_simulations.keys():
        Shared.all_simulations[base]['snewpy']={}
        print(base)
        for dec in range(len(detector_name[:3])):
            distance = 10 # Supernova distance in kpc
            detector = detector_name[dec] #"wc100kt30prct" # Name of SNOwGLoBES detector model to use
            modeltype = 'aurore_2024' # Model type from snewpy.models
            model = base # Name of model
            transformation = 'NonAdiabaticMSWH_NMO' # Desired flavor transformation
            
            # Construct file system path of model file and name of output file
            modelfile = Shared.SNEWPY_model_dir + "/" + modeltype + "/" + model[:-4]+'.txt'
            outfile = modeltype + "_" + model + "_" + transformation
            

            # There are three ways to select a time range.
            # Option 1 - don't specify tstart and tend, then the whole model is integrated
            #tstart = None
            #tend = None
            
            # Option 2 - specify single tstart and tend, this makes 1 fluence file integrated over the window
            #tstart = 0.7 * u.s
            #tend = 0.8 * u.s
            
            # Option 3 = specify sequence of time intervals, one fluence file is made for each interval
            file_start=Shared.all_simulations[base]['time'][0]
            file_end=Shared.all_simulations[base]['time'][-1]
            window_tstart = Shared.all_simulations[base]['time'][0]
            window_tend = Shared.all_simulations[base]['time'][-1]
            window_bins = 200
            tstart = np.linspace(window_tstart, window_tend, window_bins, endpoint=False) * u.s
            tend = tstart + (window_tend - window_tstart) / window_bins * u.s
            tmid = (tstart + tend) * 0.5
            # print("Preparing fluences ...")
            tarredfile = snowglobes.generate_fluence(modelfile, modeltype, transformation, distance, outfile, tstart, tend)
            
            # Next, we run SNOwGLoBES. This will loop over all the fluence files in `tarredfile`.
            # print("Running SNOwGLoBES ...")
            snowglobes.simulate(Shared.SNOwGLoBES_path, tarredfile, detector_input=detector)
            
            # Finally, we collate SNOwGLoBES’ results into a dictionary
            # print("Collating results ...")
            tables = snowglobes.collate(Shared.SNOwGLoBES_path, tarredfile, skip_plots=True)
            nevents = np.zeros(len(tmid))
            for i in range(len(tmid)):
                key = f"Collated_{outfile}_{i}_{detector}_events_smeared_weighted.dat"
                for j in range(1,len(tables[key]['header'].split())):
                    nevents[i] += sum(tables[key]['data'][j])
            
            # nevents is per bin, convert to per ms
            factor = window_bins / (window_tend - window_tstart) / 1000
            # dict[filename]=[tmid,nevents*factor]
    
            Shared.all_simulations[base]['snewpy'][detector_name[dec]]={}
            Shared.all_simulations[base]['snewpy'][detector_name[dec]]['total_events']=np.sum(nevents)
            Shared.all_simulations[base]['snewpy'][detector_name[dec]]['plots']=[tmid/u.s,nevents*factor]
            


def Nu_time_plot(time_range):
    neutrino_all_sim_all_detectors()
    
    for dec in range(len(detector_name[:3])):
        plt.figure(1)
        for base in Shared.all_simulations.keys():
            time,y=Shared.all_simulations[base]['snewpy'][detector_name[dec]]['plots']
            plt.plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])
            
        plt.xlabel('Time post-bounce [s]')
        plt.ylabel('Number of detections')
        
        plt.title(detector_name[dec])
        plt.savefig('figures/'+detector_name[dec]+'_nu_event_time.png')




def Make_snewpy_files():
    for base in Shared.all_simulations.keys():
        if os.path.isfile('snewpy_models/aurore_2024/'+base[:-4]+'.txt'):
            print("Snewpy files already done!")
            continue
        elif 'GR1D' in base:
            values=np.zeros(10)
            dump_var=np.loadtxt(Shared.GR1D_path+base[:-5]+"/"+"M1_flux_lum.dat",unpack=True,usecols=(0,1,2,3))
            values[0]=dump_var[0]
            values[1]=dump_var[1]
            values[2]=dump_var[2]
            values[3]=dump_var[3]
            dump_var=np.loadtxt(Shared.GR1D_path+base[:-5]+"/"+"M1_flux_aveenergy_lab.dat",unpack=True,usecols=(1,2,3))
            values[4]=dump_var[0]
            values[5]=dump_var[1]
            values[6]=dump_var[2]
            dump_var=np.loadtxt(Shared.GR1D_path+base[:-5]+"/"+"M1_flux_rmsenergy_lab.dat",unpack=True,usecols=(1,2,3))
            values[7]=dump_var[0]
            values[8]=dump_var[1]
            values[9]=dump_var[2]
            
            matrix=np.zeros((10,len(values[0])))
            for n in range(9):
                t,matrix[n+1]=Shared.plot_interp(values[0],values[n+1])
            matrix[0]=t
            np.savetxt('./snewpy_models/aurore_2024/'+base[:-4]+'.txt',matrix.T, header=str(Shared.all_simulations[base]['t_bounce']))

        else:
            values = np.loadtxt(Shared.FLASH_path+base,unpack=True,usecols=(0,33,34,35,36,37,38,39,40,41))
            values[1:4]=values[1:4]*1e51
            
        
            matrix=np.zeros((10,len(values[0])))
            for n in range(9):
                t,matrix[n+1]=Shared.plot_interp(values[0],values[n+1])
            matrix[0]=t
            np.savetxt('./snewpy_models/aurore_2024/'+base[:-4]+'.txt',matrix.T, header=str(Shared.all_simulations[base]['t_bounce']))

            
    
        
    