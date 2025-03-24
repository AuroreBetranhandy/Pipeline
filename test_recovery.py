#!/usr/bin/env python
# coding: utf-8

# In[1]:


import snewpy
import numpy as np
import scipy 
import matplotlib.pylab as plt
from snewpy import snowglobes
import warnings
warnings.simplefilter('ignore')
import csv
from matplotlib.lines import Line2D
SNOwGLoBES_path = "/data/home/abetranhandy/SN_stuff/snowglobes/" # directory where SNOwGLoBES is located
SNEWPY_model_dir = "snewpy_models/"
list_file= get_ipython().getoutput('cd $SNEWPY_model_dir/aurore_2024/ &&  ls s20*SFHo*8.txt  | grep -v -e "hr" -e "nr"   && ls s20*ax_on*.txt &&  ls s20*SRO*.txt             &&   ls s20*e8*.txt')
names=[]
numbers=[]
save_number=[]


# In[2]:


detec=np.loadtxt(SNOwGLoBES_path+'detector_configurations.dat',unpack=True, dtype=str)
print(detec)
detector_name=detec[0]
detector_mass=detec[1]
detector_norma=detec[2]
detector_name=np.delete(detector_name,np.argwhere(detector_name == 'halo1'))
detector_name=np.delete(detector_name,np.argwhere(detector_name == 'halo2'))

print(detector_name)
   


# In[3]:


def lets_go(filename):
    for dec in range(len(detector_name[:3])):
        distance = 10 # Supernova distance in kpc
        detector = detector_name[dec] #"wc100kt30prct" # Name of SNOwGLoBES detector model to use
        modeltype = 'aurore_2024' # Model type from snewpy.models
        model = filename # Name of model
        transformation = 'NonAdiabaticMSWH_NMO' # Desired flavor transformation
        
        # Construct file system path of model file and name of output file
        model_path = SNEWPY_model_dir + "/" + modeltype + "/" + model
        outfile = modeltype + "_" + model + "_" + transformation
        
        # Now, do the main work:
        print("Generating fluence files ...")
        tarredfile = snowglobes.generate_fluence(model_path, modeltype, transformation, distance, outfile)
        
        print("Simulating detector effects with "+detector)
        snowglobes.simulate(SNOwGLoBES_path, tarredfile, detector_input = detector)
        
        print("Collating results ...")
        tables = snowglobes.collate(SNOwGLoBES_path, tarredfile, detector_input = detector,skip_plots = False)
        # Use results to print the number of events in different interaction channels
        key = f"Collated_{outfile}_{detector}_events_smeared_weighted.dat"
        total_events = 0
        for i, channel in enumerate(tables[key]['header'].split()):
            if i == 0:
                continue
        # Scale to Super-K inner volume (32 kt)
        n_events =  sum(tables[key]['data'][i])
        total_events += n_events
        print(f"{channel:10}: {n_events:.3f} events")
        
        print("Total events"+ str(i) + detector_name[dec] +" detector:", total_events)
        globals()[filename+'number'+detector]=total_events
        # numbers.append(total_event
        save_number.append([filename,detector_name[dec],total_events])


# In[4]:


for i in list_file:
    # print(i)
    if ('SFHo' in i ) & ('8.dat' in i) :
        globals()['alpha'+i]=1.
    elif  ('SFHo' in i ) & ( '7.dat' in i) :
        globals()['alpha'+i]=0
    elif ('SRO' in i) & ('8' in i) :
        globals()['alpha'+i]=0.
    elif  ('SRO' in i)  :
        globals()['alpha'+i]=1.
    else: 
        globals()['alpha'+i]=1.
        
    if 'ref' in i: 
        globals()['ticks'+i]='o'
    elif 'e8' in i: 
        globals()['ticks'+i]='o'
    else:
        globals()['ticks'+i]='^'
        
        
    if ('e8' in i) or ('ax_on' in i):
        globals()['tb'+i]= 0.319
        globals()['alpha'+i]=1.
        col = plt.cm.jet(np.linspace(1,0,7))
        globals()['ticks'+i]="D"
        if 'ref' in i: 
            globals()['color'+i]=col[5]
        elif '1_5e8' in i:
            globals()['color'+i]=col[1]
        elif '1e8' in i:
            globals()['color'+i]=col[0]
        elif '2e8' in i:
            globals()['color'+i]=col[2]
        elif '3e8' in i:
            globals()['color'+i]=col[3]
        elif '5e8' in i:
            globals()['color'+i]=col[4]
            
    elif ('Gang' in i) & ('SFHo' in i) : 
        globals()['color'+i]='orange'
        globals()['tb'+i]=0.300
    elif 'nr' in i :
        globals()['color'+i]='m'
    elif 'SFHo' in i :
        globals()['color'+i]='r'
        globals()['tb'+i]=0.300
    elif ('Gang' in i) & ('SRO' in i) : 
        globals()['color'+i]='g'
        globals()['tb'+i]= 0.319
    elif 'SRO' in i :
        globals()['color'+i]='b'
        globals()['tb'+i]= 0.319
      
    else:
        globals()['color'+i]='k'
        globals()['tb'+i]= 0.319
        print("case not found")
        


# In[5]:


for i in list_file: 
    print(i)
   
    # try:
    lets_go(i)
    # except:
        # continue


# In[6]:


# expl_time={'s20_ref_Gang_SFHo8.dat': 0.2839332441563451, 's20_ref_Gang_SFHo_new8.dat': 0.2582226248545974, 's20_ref_Hann_SFHo8.dat': 0.3288730922369449, 's20_ref_SFHo_nb78.dat': 0.3121256452833346, 's20_simp_SFHo_Gang88.dat': 0.46383709458964634, 's20_simp_SFHo_Hann88.dat': 0.42905157743334826, 's20_ref_ax_on.dat': 0.6821663287137285, 's20_ref_Gang_SRO.dat': 0.93766057878735, 's20_ref_SRO3.dat': 0.9679960508300613, 's20_simp_SRO3.dat': 1.012863151949718, 's20_simp_SRO_Gang8.dat': np.nan, 's20_simp_SRO_Gang.dat': 0.7428695786142567, 's20_simp_SRO_Gang_old.dat': 0.7428695786142567, 's20_1_5e8.dat': 0.3526906648360321, 's20_1e8.dat': 0.2666621626951136, 's20_2e8.dat': 0.2705053379435353, 's20_3e8.dat': 0.6778566393911833, 's20_5e8.dat': 0.8120019891956711}
# expl_time={'s20_ref_Gang_SFHo8.dat': 0.3672, 's20_ref_Gang_SFHo_new8.dat': 0.3108, 's20_ref_Hann_SFHo8.dat': 0.3894, 's20_ref_SFHo_nb78.dat': 0.3974, 's20_simp_SFHo_Gang88.dat': 0.5153, 's20_simp_SFHo_Hann88.dat': 0.4965, 's20_ref_ax_on.dat': 0.7373, 's20_ref_Gang_SRO.dat': 0.9376, 's20_ref_SRO3.dat': 0.9680, 's20_simp_SRO3.dat': 1.0129, 's20_simp_SRO_Gang8.dat': np.nan, 's20_simp_SRO_Gang.dat': 0.7429, 's20_simp_SRO_Gang_old.dat': 0.7429, 's20_1_5e8.dat': 0.4131, 's20_1e8.dat': 0.3229, 's20_2e8.dat': 0.4303, 's20_3e8.dat': 0.6988, 's20_5e8.dat': 0.8485}
expl_time={'s20_ref_Gang_SFHo8.dat': 0.3672, 's20_ref_Gang_SFHo_new8.dat': 0.3109, 's20_ref_Hann_SFHo8.dat': 0.3894, 's20_ref_SFHo_nb78.dat': 0.3974, 's20_simp_SFHo_Gang88.dat': 0.5153, 's20_simp_SFHo_Hann88.dat': 0.4965, 's20_ref_ax_on.dat': 0.7373, 's20_ref_Gang_SRO.dat': 0.9377, 's20_ref_SRO3.dat': 0.968, 's20_simp_SRO3.dat': 1.0406, 's20_simp_SRO_Gang.dat': 0.789, 's20_simp_SRO_Gang8.dat': np.nan, 's20_simp_SRO_Gang_old.dat': 0.789, 's20_1_5e8.dat': 0.4131, 's20_1e8.dat': 0.3229, 's20_2e8.dat': 0.4302, 's20_3e8.dat': 0.6988, 's20_5e8.dat': 0.8485}
sorted_names={k: v for k, v in sorted(expl_time.items(), key=lambda item: item[1])}


# In[7]:


print(save_number)


# In[11]:


# plt.plot(numbers,names,'o'),
num_of_dec=2
fig,ax = plt.subplots(num_of_dec,1,figsize=(10,5*num_of_dec)) #figsize=(10,5*len(detector_name[0:6])))
print(ax.shape)

for dec in range(num_of_dec):
    ax[dec].set_title(detector_name[dec])
    ax[dec].set_ylim([0,600])
    for name_dat in sorted_names.keys():
        i=name_dat[:-4]+'.txt'
        # print(globals()[i+'number'+detector_name[dec]])
        try:
            ax[dec].plot(sorted_names[name_dat],globals()[i+'number'+detector_name[dec]],color=globals()['color'+i],marker=globals()['ticks'+i],alpha=globals()['alpha'+i])
        except:
            continue
# for dec in range(num_of_dec-1):
    # ax[dec].set_title(detector_name[dec])
    # ax[dec].set_xticklabels([])    
# ax[-1].set_xticks(range(len(sorted_names.keys())), sorted_names.keys(),rotation=90,fontsize=10)
# ax[-1].set_xticks(range(len(sorted_names.keys())), sorted_names.values(),rotation=90,fontsize=10)
# ax[-1].set_xticks(range(len(sorted_names.keys())), expl_time.values(),rotation=90,fontsize=10)



black_line1, = ax[1].plot([], [], color='b', linestyle='-')
black_line2, = ax[1].plot([], [], color='g', linestyle='-')
leg4=ax[1].legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=14,handlelength=1,title='SFHo',title_fontsize=16,frameon=False,loc='upper right')
                 
black_line1, = ax[1].plot([], [], color='r', linestyle='-')
black_line2, = ax[1].plot([], [], color='orange', linestyle='-')
leg3=ax[1].legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=14,handlelength=1,title='SFHo',title_fontsize=16,frameon=False,loc='upper left')

l1, = ax[0].plot([], [], color=col[0])
l2, = ax[0].plot([], [], color=col[1])
l3, = ax[0].plot([], [], color=col[2])
l4, = ax[0].plot([], [], color=col[3])
l5, = ax[0].plot([], [], color=col[4])
l6, = ax[0].plot([], [], color=col[5])
leg1 =ax[0].legend([l1,l2,l3,l4,l5,l6],[r"1 $\times$ 10$^8$ GeV",r"1.5 $\times$ 10$^8$ GeV",r"2 $\times$ 10$^8$ GeV",\
                                        r"3 $\times$ 10$^8$ GeV",r"5 $\times$ 10$^8$ GeV","reference"]\
                    ,loc='lower right',handlelength=1.1,fontsize=14,title=r'f_a ',title_fontsize=16,frameon=False)

black_line1, = ax[0].plot([], [], color='w', marker='o',markerfacecolor='k',markersize=8)
black_line2, = ax[0].plot([], [], color='w', marker='^',markerfacecolor='k',markersize=8)
black_line3, = ax[0].plot([], [], color='w', marker='D',markerfacecolor='k',markersize=8)
leg2=ax[0].legend([black_line1,black_line2,black_line3],[r"Reference",r"$\kappa^*$","Axions"],fontsize=16,handlelength=1,frameon=False)

ax[1].set_xlabel('t [s]',fontsize=20)
ax[1].set_ylabel(r'Total $\nu$ count',fontsize=20)
ax[0].set_ylabel(r'Total $\nu$ count',fontsize=20)
ax[0].add_artist(leg1)
ax[1].add_artist(leg3)
ax[1].add_artist(leg4)
# plt.subplots_adjust(wspace=0,hspace=0.)

ax[0].tick_params(axis='both', which='major', labelsize=18,width=2,length=4)
ax[0].tick_params(axis='both', which='minor', labelsize=18,width=1.5,length=2)
ax[1].tick_params(axis='both', which='major', labelsize=18,width=2,length=4)
ax[1].tick_params(axis='both', which='minor', labelsize=18,width=1.5,length=2)
plt.legend(loc='center left', bbox_to_anchor=(1, 1))


# In[ ]:





# In[ ]:




