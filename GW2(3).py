#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pandas as pd
                                                                                                                          
matplotlib.rcParams['xtick.direction'] = 'in'                                                                                               
matplotlib.rcParams['ytick.direction'] = 'in'                                                                                               
matplotlib.rcParams['xtick.top'] = True                                                                                                     
matplotlib.rcParams['ytick.right'] = True                                                                                                   
                                                                                                                                            
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'                                                                                           
matplotlib.rcParams.update({'pgf.rcfonts' : False})                                                                                         
                                                                                                                                            
matplotlib.rcParams['axes.linewidth'] = 2                                                                                                   
matplotlib.rcParams['xtick.major.size'] = 6                                                                                                
matplotlib.rcParams['xtick.major.width'] = 2                                                                                                
matplotlib.rcParams['xtick.minor.size'] = 4                                                                                                
matplotlib.rcParams['xtick.minor.width'] = 0.5                                                                                                
                                                                                                                                            
matplotlib.rcParams['ytick.major.size'] = 6                                                                                                
matplotlib.rcParams['ytick.major.width'] = 2                                                                                                
matplotlib.rcParams['ytick.minor.size'] = 4                                                                                                
matplotlib.rcParams['ytick.minor.width'] = 0.5

plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

plt.rcParams['axes.linewidth'] = 3.
plt.rcParams['xtick.major.width'] = 1.5
#plt.rcdefaults()
    
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def dataframe(filename):
    with open(filename, 'r') as file:
        first_line = file.readline()

    # Now we'll create a regex that searches for the patterns in the form of:
    # (number)(spaces)(any characters that is not a digit)(lookahead for the next number or the end of line)
    column_names = []
    for i in range(1, 100):  # Assuming we have less than 100 columns; adjust if there are more.
        # The following regex looks for a specific number (i), followed by a space, and captures all characters until the next number
        pattern = rf"(?<=\b{i}\s)(.*?)(?=\s+\b{i+1}\b|$)"
        match = re.search(pattern, first_line)
        if match:
            # Add the found column name to the list after stripping leading/trailing whitespaces
            column_names.append(match.group(0).strip())
        else:
            # Break the loop if a number is not found, assuming we've found all columns
            break

    df = pd.read_csv(filename,names=column_names, delim_whitespace=True, low_memory=False,skiprows=1)
    return df


# In[2]:


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

def read_models(model_file,iskip=10):
    with open(model_file) as f_in:
        model = np.genfromtxt(itertools.islice(f_in, 0, None, iskip), dtype=float)
    return model

def find_bounce(file_path):
    if 'SFHo' in file_path:
        return 0.299
    if ("SRO" in file_path) or ("e8" in file_path) or ('ax_on' in file_path):
        return 0.319
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         if 'Bounce!' in line:
    #             return line.strip()  # Return the line without leading/trailing whitespace
    # return  

def spectrogram_full(t,h,shift_value=1e-4,window_length=25e-3):
    from scipy.fftpack import fft
    from scipy import signal
    N = 4*512
    nf = 2*N
    wdt = window_length
    dt = wdt/(nf)
    freq = np.fft.fftfreq(nf, d=dt)

    ffpeuq = interp1d(t, h)
    tshift = min(t)
    fnyq = max(freq)
    bl, al = signal.butter(2, 5000.0/fnyq,'low')
    bh, ah = signal.butter(2, 25.0/fnyq,'high')
    window = signal.kaiser(nf,2.5)
    window =  signal.blackman(nf)
    t2 = np.linspace(tshift,wdt+tshift,nf)
    tarr = []
    fttm = []
    j = 0
    while(max(t2) < max(t)):
        amp = ffpeuq(t2)
        amp1 = signal.filtfilt(bh, ah, amp)
        amp2 = signal.filtfilt(bl, al, amp1)
        amp3 = amp2*window
        fftamp  = (np.abs(fft(amp3))[0:N]**2) /np.sum(window)**2
        fttm.append([])
        fttm[j] = fftamp
        tarr.append((max(t2)+min(t2))*0.5)
        tshift = tshift + shift_value
        t2 = np.linspace(tshift,wdt+tshift,nf)
        j = j+1
    rt = fttm,freq[0:N],tarr
    return rt

def plot_spectrogram(t,h,filename="specs.png",modes=None,cmap='viridis',
                    shift_value=1e-4,window_length=25e-3,nlevels=50,low=-6,high=0):
    t = t - min(t)
    jet = cm = plt.get_cmap(cmap)
    v = np.linspace(low,high,nlevels)
    
    f1,ax=plt.subplots(figsize=(20,10),ncols=1,nrows=1,sharex="row",sharey="row")
    ffc,f,t1=spectrogram_full(t,h,shift_value,window_length)
    ft=np.log10((np.transpose(ffc/np.max(ffc))**2+1e-13))
    cb=ax.contourf(np.array(t1),f,ft,v,cmap=jet,extend="both")
    f1.subplots_adjust(hspace=0.,wspace=0.0)
    ax.set_ylim([25,2250])
    ax.set_xlim([min(t1),max(t1)])
    ax.set_yticks([100,250,400,550,700,850,1000,1150])
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))
    ax.set_xlabel(r'$\mathrm{Time\,[ms]}$')
    ax.set_ylabel(r'$\mathrm{Frequency\,[Hz]}$')
    
    if(modes):
        for key in modes.keys():
            ax.plot(t,modes[key],'--',lw=3.5,c="black")

    f1.colorbar(cb,ax=ax,ticks=[0,-1,-2,-3,-4,-5,-6],pad=0.007,aspect=30)
    plt.savefig(filename,bbox_inches='tight')

def plot_signal(t,hx,hp,filename="amps.png"):
    f1,ax=plt.subplots(figsize=(20,10),ncols=1,nrows=2,sharex="row",sharey="row")
    f1.subplots_adjust(hspace=0.,wspace=0.0)
    ax[0].plot(t,hx)
    ax[0].set_ylim([-1.5,1.5])
    ax[0].set_xlim([min(t),max(t)])
    ax[0].set_yticks([-1,-0.5,0,0.5,1])
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(3))
    ax[0].set_ylabel(r'$\mathrm{Amplitude}$') 
    ax[1].set_xlabel(r'$\mathrm{Time\,[ms]}$')
    
    ax[1].plot(t,hp)
    ax[1].set_ylim([-1.5,1.5])
    ax[1].set_xlim([min(t),max(t)])
    ax[1].set_yticks([-1,-0.5,0,0.5,1])
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(3))
    ax[1].set_ylabel(r'$\mathrm{Amplitude}$') 
    plt.savefig(filename,bbox_inches='tight')



def spectrogram(I2_interp, time_list, delta_t_mean, sample_frequency_mean, 
                hann_window=40e-3, overlap_fact=0.5, nfft_fact=1, name=None, plot=False, energy=False,
               window_name = 'hann'):
    if energy == False:
        Nperseg = int(hann_window/delta_t_mean)
        Noverlap = int(Nperseg*overlap_fact)
        Nfft = Nperseg*nfft_fact
        #print(len(time_list)/Nperseg)
        prefactor = (3/2)*(cn.G/cn.c**4).cgs.value

        t2 = np.linspace(time_list[0],time_list[-1],len(time_list)*nfft_fact)
        dt = t2[1]-t2[0]
        df = 1/dt
        fnyq = 0.5*df
        I2_eq = I2_interp(t2)
        
        bl, al = signal.butter(2, 3000.0/fnyq,'low')
        bh, ah = signal.butter(2, 25.0/fnyq,'high')

        I2_eq = signal.filtfilt(bh, ah,  I2_eq)
        I2_eq= signal.filtfilt(bl, al,  I2_eq)
        
        f_list, tau_list, Zxx = stft(prefactor*I2_eq,df, 
                                     nperseg=Nperseg, noverlap=Noverlap, window=window_name, nfft=Nfft)
        spectrogr = np.abs(Zxx)**2
        
    if energy:
        Nperseg = int(hann_window/delta_t_mean)
        Nfft = Nperseg*nfft_fact
        #print(len(time_list)/Nperseg)
        f_list, tau_list, Zxx = stft(I2_interp(time_list), sample_frequency_mean, 
                                     nperseg=Nperseg,window=window_name, nfft=Nfft)
        Zabs = np.abs(Zxx)
        #print(Zxx.shape)
        
        pref_matrix = np.zeros((len(f_list),len(tau_list)))
        Constant = 3/5 *(cn.G/cn.c**5).cgs.value*(2*np.pi)**2
        for i in range(len(f_list)):
            for j in range(len(tau_list)):
                pref_matrix[i][j] = Constant*f_list[i]**2
                #print(pref_matrix[i][j])
        spectrogr = np.multiply(Zabs, pref_matrix)
        
    if plot:
        plt.figure(figsize=(15,8))
        v = np.linspace(-6,0.1,70)
        plt.contourf(tau_list, f_list, np.log10(np.abs(spectrogr)**2/3e3+1e-13),
                     v,extend="min")
        print(np.max(np.abs(spectrogr)**2))
        plt.colorbar(orientation='horizontal')
        plt.ylim(0,2000)
        #plt.xlim(0,0.4)
    return f_list, tau_list, spectrogr


# In[4]:


# mdlz85 = read_models('/proj/astro_extreme/haakon/master_projects/li/simulations/z85SFHx/z85_SFHx_3e9.dat')
# mdlz85check = read_models('/proj/astro_extreme/haakon/master_projects/li/simulations/z85SFHx/z85_SFHx_gwcheck.dat')
list_files=get_ipython().getoutput('ls s20*SFHo*8.dat  | grep -v -e "hr" -e "nr"   && ls s20*ax_on*.dat &&  ls s20*SRO*.dat &&   ls s20*e8*.dat')
# substrings = ["_pf", "he","gitGR1D","_PCC","_huth"]
mdlz85check_list= [read_models(x) for x in list_files]


# In[5]:


# list_files= ['s20_ref_Gang_SFHo_hr.dat']

# print(list_files)
# !ls -t *.npy
# list_files.remove('s20_simp_SRO_Gang8.dat')
# list_files.remove('s20_simp_SFHo_Gang.dat')
# list_files.remove('s20_ref_Hann_SFHo_hr.dat')
# list_files.remove("s20_ref_Gang_SFHo_per.dat")
# list_files.append('s20_ref_ax_on.dat')
x=0
for i in list_files:
    if 'SFHo' in i:
        if not exists('allprofiles'+i[:-5]+'.npy'):
            list_files.remove(i)
            continue

        globals()[i]=np.load('allprofiles'+i[:-5]+'.npy',allow_pickle=True).item()
    else:
        if not exists('allprofiles'+i[:-4]+'.npy'):
            list_files.remove(i)
            continue

        globals()[i]=np.load('allprofiles'+i[:-4]+'.npy',allow_pickle=True).item()


# In[24]:


mdl= 's20_simp_SRO3.dat'
# print(globals().keys()['SRO' in x for x in globals().keys()])
for i in globals().keys():
    if 'SRO' in i:
        print(i)
time_N2,kin_erg=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["kin_erg"],500)


# In[ ]:


import scipy
tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
(freq, Sigs) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')

f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')

ax.plot(freq, Sigs)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
ax.set_ylim([0,0.15e98])
ax.set_xlim([0,500])


# In[ ]:





# In[10]:


aurore_models_dir = get_ipython().getoutput('ls s20*SFHo*8.dat  | grep -v -e "hr" -e "nr"   && ls s20*ax_on*.dat &&  ls s20*SRO*.dat &&   ls s20*e8*.dat')
# substrings = ["_pf", "he","gitGR1D","_PCC","_huth"]


for i in aurore_models_dir:
    # print(i)
    globals()[i[:]+"shock"] = np.loadtxt(i,unpack=True,usecols=(0,11,18,30))
factor=1e-5

for i in aurore_models_dir:
    # print(i)
    
    print(i)
    tau=globals()[i+"shock"][1]
    x=globals()[i+"shock"][0]
    time,tau_new=plot_interp_minimize(x,tau,100)
    plt.plot(time,tau_new*factor,color=globals()['color'+i],ls=globals()['ticks'+i])
   
    try:
        if ('SRO' in i) and ("ref" in i):
            globals()['time_explo'+i]=round((time[tau_new*factor>200][0]-find_bounce(i)),4)
        else:    
            globals()['time_explo'+i]=round((time[tau_new*factor>400][0]-find_bounce(i)),4)
        
    except:
        globals()['time_explo'+i]=np.nan
    print(globals()['time_explo'+i])
plt.hlines(400,0,1.5)
plt.hlines(200,0,1.5)
# evan_models_dir = [item for item in aurore_models_dir if all(sub not in item for sub in substrings)]
evan_models = {}
bt_em = {}
expl_time = {}
for item in aurore_models_dir:
    modl_name=item
    path = item
    print(path,modl_name)
    evan_models[modl_name] = read_models(path,iskip=1)
    
    path = f'{modl_name}'
    bt_em[modl_name] = find_bounce(path)
    expl_time[modl_name] = globals()['time_explo'+modl_name]
    
    
# bt_g10s = {}
# bt_g10s['s11'] = 0.05*4 - bt_em['s11']
# bt_g10s['s12'] = 0.05*4 - bt_em['s12']
# bt_g10s['s13'] = 0.05*5 - bt_em['s13']
# bt_g10s['s14'] = 0.05*6 - bt_em['s14']
# bt_g10s['s15'] = 0.05*5 - bt_em['s15']


# In[11]:


print(expl_time)
sorted_names={k: v for k, v in sorted(expl_time.items(), key=lambda item: item[1])}
print(sorted_names.keys())

with open('time_dependent_nu_event.pkl', 'rb') as f:
    nu_events = pickle.load(f)


# In[8]:


for i in aurore_models_dir:
    # print(i)
    if ('SFHo' in i ) & ('8.dat' in i) :
        globals()['alpha'+i]=1.
    elif  ('SFHo' in i ) & ( '7.dat' in i) :
        globals()['alpha'+i]=0
    elif ('SRO' in i) & ('8' in i) :
        globals()['alpha'+i]=0.
    elif  ('SRO' in i)  :
        globals()['alpha'+i]=1.
    elif ('SFHo' in i) & (('per' in i) or ('hr' in i)) :
        globals()['alpha'+i]=0.
    else: 
        globals()['alpha'+i]=1.
        
    if 'ref' in i: 
        globals()['ticks'+i]='-'
    elif 'e8' in i: 
        globals()['ticks'+i]='-'
    else:
        globals()['ticks'+i]='--'
        
    if 'ref' in i: 
        globals()['marker'+i]='o'
    elif 'e8' in i or 'ax_on' in i: 
        globals()['marker'+i]='D'
    else:
        globals()['marker'+i]='^'
        
    if ('e8' in i) or ('ax_on' in i):
        globals()['tb'+i]= 0.319
        col = plt.cm.jet(np.linspace(1,0,7))
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
        


# In[54]:


# mdl = 's12'
# first_index = np.where(evan_models[mdl][:,0] > bt_em[mdl])[0][0]#this should be bounce time
# last_index = np.where(evan_models[mdl][first_index:,0] - bt_em[mdl] >HAR06K095_grey_models[mdl][-1,0]+bt_g10s[mdl] )[0][0]
prefactor = (3/2)*(cn.G/cn.c**4).cgs.value 
    #first_index = 0
    # Extract the 'time' column as an array
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# sorted_names.pop('s20_simp_SRO_Gang_old.dat')
# sorted_names.pop('s20_simp_SRO_Gang8.dat')


figure,ax_comp= plt.subplots(figsize=(5*len(evan_models.keys())+5,20),nrows=10,ncols=len(sorted_names.keys()),sharey='row')
number=0

for mdl in sorted_names.keys():
    time_nu,nevent=nu_events[mdl[:-4]+'.txt']
    if 'hr' in mdl:
        continue
    time = evan_models[mdl][:,0]-bt_em[mdl] # this time should be bounce time
    strain = evan_models[mdl][:,59]+evan_models[mdl][:,53]+evan_models[mdl][:,47]

    timecheck1,straincheck1=plot_interp(time,strain)
    timecheck=timecheck1[timecheck1>0]
    straincheck=straincheck1[timecheck1>0]
    straincheck = np.gradient(straincheck,timecheck)
    
    # f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
    #ax.plot(time,strain)
    # ax.plot(timecheck,straincheck*prefactor)
    # ax.set_xlabel("Time post bounce")
    #ax.set_xlim([-0.01,0.6])
    # ax.set_ylim([-40,40])
    
    
    
    fiper = interpolate.interp1d(timecheck,straincheck*prefactor)
    tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
    (freqed, Sigsed) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')
    
    
    
    
    fiper = interpolate.interp1d(timecheck,straincheck)
    
    delta_t_mean = np.mean(np.diff(timecheck))
    sample_frequency_mean = 1/delta_t_mean
    
    f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                              hann_window=35e-3, overlap_fact=0.95, nfft_fact=1, 
                                              name=None, plot=False, energy=False,
                                             window_name='blackman')

    time_reduc,strain_reduc=plot_interp_minimize(timecheck,straincheck,int(800*timecheck.max()))
    ax_comp[0,number].plot(time_reduc,strain_reduc*prefactor,lw=2,color=globals()['color'+mdl],ls=globals()['ticks'+mdl])
    # ax[0,1].plot(timecheckHA,straincheckHA*prefactor,lw=2)
    
    v = np.linspace(-3,0,40)
    # ax[1,0].contourf(tau_list, f_list, np.log10(np.abs(spectrogr)/15),v,extend="both")
    cba = ax_comp[1,number].contourf(tau_list, f_list, np.log10(np.abs(spectrogr)/15),v,extend="both")
    ax_comp[1,number].set_ylim([0,2500])
    ax_comp[2,number].set_ylim([0,700])
    ax_comp[1,number].set_xlim([0.01,1.09])
    ax_comp[0,number].set_xlim([0.01,1.09])
    ax_comp[2,number].set_xlim([0.01,1.09])
    #ax[1,1].axhline(1100,c='white',ls="--",lw=2)
    #ax[1,0].axhline(1100,c='white',ls="--",lw=2)
    ax_comp[1,number].set_yticks([300,600,900,1200,1500,1800,2100,2400])
    ax_comp[0,number].set_yticks([-30,-15,0,15,30])
    ax_comp[0,number].set_ylim([-35,35])
    # ax_comp[1,number].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])


#cb = plt.colorbar(cba,ax=ax[1,1],location='right')
    ax_comp[0,number].set_title(mdl+" tb="+str(expl_time[mdl])[:5],fontsize=12)
    
    tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
    (freq, Sigs) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')
    
    ax_comp[2,number].set_xlabel("Time [s]")


    freq_reduc,Sigs_reduc=plot_interp_minimize(freq, Sigs,7000)
    time_reduc,straincheck_reduc=plot_interp_minimize(timecheck, straincheck,7000)

    
    A_square=fft(straincheck)
    freq_space = np.linspace(0.0, 1.0/(2.0*delta_t_mean), A_square.size//2)
    dE_df= 3/5 * (cn.G.cgs.value/cn.c.cgs.value**5) * (2*np.pi*freq_space)**2 * np.abs(A_square[:straincheck.size//2]*2/straincheck.size)**2
    h_char=1/(10*cn.pc.cgs.value*1e3) * np.sqrt((2/np.pi**2) * (cn.G.cgs.value/cn.c.cgs.value**3) * dE_df)

    freq_space_reduc,h_char_reduc=plot_interp_minimize(freq_space, h_char,2000)

    
    # freq_reduc,Sigs_reduc=plot_interp_minimize(freq, Sigs,5000)
    ax_comp[3,number].loglog(freq_space_reduc,h_char_reduc,color=globals()['color'+mdl],ls=globals()['ticks'+mdl] )
    ax_comp[3,number].set_xlabel('frequency [Hz]')
    # ax_comp[3,number].set_ylim([0,0.15e98])
    # ax_comp[3,number].set_xlim([0,2400])
    ax_comp[3,number].set_xlim([3e2,1e4])
    ax_comp[3,number].set_ylim(1e-24,1e-19)

    # fixed_nuevent= np.interp1d(time_reduc,time_fucking,event_fucking)
    ax_comp[2,number].plot(time_nu,nevent ,color=globals()['color'+mdl],ls=globals()['ticks'+mdl] )

   
    try:
        time_N2,kin_erg=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["kin_erg"],500)
        time_N2,conv_max_rad=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["conv_max_rad"],500)
        time_N2,conv_min_rad=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["conv_min_rad"],500)
        time_N2,total_mass=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["total_mass"],500)
        time_N2,kinetic_shock=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["kinetic_shock"],100)
        time_N2,r_shock_in=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["r_shock_in"],500)
        time_N2,r_shock_out=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["r_shock_out"],500)
        time_N2,mass_shock=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["mass_shock"],100)
    
    
    
        
        globals()[mdl]["conv_max_rad"][globals()[mdl]["conv_max_rad"]==0.]=np.nan
        globals()[mdl]["kin_erg"][globals()[mdl]["kin_erg"]==0.]=np.nan
        
        ax_comp[5,number].plot(time_N2,kin_erg/1e49,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])    
        ax_comp[4,number].plot(time_N2,conv_max_rad,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])        
        ax_comp[4,number].plot(time_N2,conv_min_rad,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])
        ax_comp[6,number].plot(time_N2,total_mass/2e33,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])    
    
    
    
        ax_comp[8,number].plot(time_N2,kinetic_shock/1e49,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])    
        ax_comp[7,number].plot(time_N2,r_shock_in,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])        
        ax_comp[7,number].plot(time_N2,r_shock_out,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])
        ax_comp[9,number].semilogy(time_N2,mass_shock/2e33,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl])    

    except:
        print(mdl)

    ax_comp[6,number].set_xlabel('t$_{pb}$ [s]',fontsize=19)

    ax_comp[4,number].set_xlim([0.01,1.09])
    ax_comp[5,number].set_xlim([0.01,1.09])
    ax_comp[6,number].set_xlim([0.01,1.09])
    ax_comp[5,number].set_ylim([0.01,1.19])
    ax_comp[4,number].set_ylim([0,50])
    ax_comp[6,number].set_ylim([0,0.6])

    ax_comp[7,number].set_xlim([0.01,1.09])
    ax_comp[8,number].set_xlim([0.01,1.09])
    ax_comp[9,number].set_xlim([0.01,1.09])
    ax_comp[7,number].set_ylim([0.01,150])
    ax_comp[8,number].set_ylim([0.01,4])
    ax_comp[9,number].set_ylim([0,0.3])

    
    number=number+1
figure.colorbar(cba, ax=ax_comp[:,-1], location='right',ticks=[-3,-2,-1,0],pad=0.01)
plt.minorticks_on()
ax_comp[0,0].set_ylabel(r"$\mathrm{h}_x \mathrm{D}$ [cm]")
ax_comp[1,0].set_ylabel(r"$\mathrm{Frequency}$ [Hz] ")
ax_comp[3,0].set_ylabel(r"'PSD [V**2/Hz]'")
ax_comp[2,0].set_ylabel(r"$\nu$ count")
ax_comp[4,0].set_ylabel('R$_{N^2<0}$ [km]',fontsize=20)
ax_comp[5,0].set_ylabel(r'$E_{kin,N^2<0}$' +'\n'+' [x10$^{49}$ erg]',fontsize=19)
ax_comp[6,0].set_ylabel('M$_{PNS,N^2<0}$ [M$_{\odot}$]',fontsize=19)
ax_comp[7,0].set_ylabel('R$_{gain}$ [km]',fontsize=20)
ax_comp[8,0].set_ylabel(r'$E_{kin,gain}$' +'\n'+' [x10$^{49}$ erg]',fontsize=19)
ax_comp[9,0].set_ylabel('M$_{shock,gain}$ [M$_{\odot}$]',fontsize=19)



figure.savefig('gws_SRO.pdf',bbox_inches='tight')
figure.savefig('gws_SFHo.png',bbox_inches='tight')
plt.subplots_adjust(wspace=0)



# In[60]:


# mdl = 's12'
# first_index = np.where(evan_models[mdl][:] > bt_em[mdl])[0][0]#this should be bounce time
# last_index = np.where(evan_models[mdl][first_index:] - bt_em[mdl] >HAR06K095_grey_models[mdl][-1]+bt_g10s[mdl] )[0][0]

LIGO=np.loadtxt('Ligo_04.txt',unpack=True,usecols=(0,1))
KAGRA=np.loadtxt('Kagra_25.txt',unpack=True,usecols=(0,1))
prefactor = (3/2)*(cn.G/cn.c**4).cgs.value 
    #first_index = 0
    # Extract the 'time' column as an array
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib as mpl
# mpl.rcParams['lines.linewidth'] = 3
fig,ax_comp= plt.subplots(figsize=(25,20),nrows=10,ncols=1)
figure,ax= plt.subplots(figsize=(25,20),nrows=3,ncols=3)
ax_comp[0]=ax[0,0]
ax_comp[1]=ax[0,1]
ax_comp[2]=ax[0,2]
ax_comp[3]=ax[1,0]
ax_comp[4]=ax[1,1]
ax_comp[5]=ax[1,2]
ax_comp[6]=ax[2,0]
ax_comp[7]=ax[2,1]
ax_comp[8]=ax[2,2]
# dic=np.load('spectro_fit.npy',allow_pickle=True)
# print(dic["s20_simp_SRO3.dat"])

number=0
for mdl in reversed(sorted_names.keys()):
    # if ('e8' in mdl) or ('ax_on' in mdl) or ('SFHo' in mdl) or (('SRO' in mdl) and not ('8' in mdl)) :
    #     continue
    if ('SRO' in mdl) and ('8' in mdl):
        continue

    time_nu,nevent=nu_events[mdl[:-4]+'.txt']
    if 'hr' in mdl:
        continue
    time = evan_models[mdl][:,0]-bt_em[mdl] # this time should be bounce time
    strain = evan_models[mdl][:,59]+evan_models[mdl][:,53]+evan_models[mdl][:,47]

    timecheck1,straincheck1=plot_interp(time,strain)
    timecheck=timecheck1[timecheck1>0]
    straincheck=straincheck1[timecheck1>0]
    straincheck = np.gradient(straincheck,timecheck)
    
    # f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
    #ax.plot(time,strain)
    # ax.plot(timecheck,straincheck*prefactor)
    # ax.set_xlabel("Time post bounce")
    #ax.set_xlim([-0.01,0.6])
    # ax.set_ylim([-40,40])
    

    
    
    
    fiper = interpolate.interp1d(timecheck,straincheck*prefactor)
    tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
    (freqed, Sigsed) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')
    
    
    
    
    fiper = interpolate.interp1d(timecheck,straincheck)
    
    delta_t_mean = np.mean(np.diff(timecheck))
    sample_frequency_mean = 1/delta_t_mean
    
    f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                              hann_window=35e-3, overlap_fact=0.95, nfft_fact=1, 
                                              name=None, plot=False, energy=False,
                                             window_name='blackman')

    time_reduc,strain_reduc=plot_interp_minimize(timecheck,straincheck,int(800*timecheck.max()))
    ax_comp[0].plot(time_reduc,savgol_filter(strain_reduc*prefactor,20,3),lw=0.7,color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.5)
    time_fit,freq_fit=dic[mdl]
    ax_comp[0].plot(time_fit,freq_fit,lw=1,color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=globals()['alpha'+mdl])



    
    v = np.linspace(-3,0,40)
    # ax[1].contourf(tau_list, f_list, np.log10(np.abs(spectrogr)/15),v,extend="both")
    # cba = ax_comp[0].contourf(tau_list, f_list, np.log10(np.abs(spectrogr)/15),v,extend="both",alpha=0.1)

    #ax[1,1].axhline(1100,c='white',ls="--",lw=2)
    #ax[1].axhline(1100,c='white',ls="--",lw=2)
    # ax_comp[0].set_yticks([-30,-15,0,15,30])
    # ax_comp[0].set_ylim([-35,35])
    # ax_comp[1].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])

    test=np.log10(np.abs(spectrogr)/15)
    test[test<-1.3]=np.nan

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none",globals()['color'+mdl]])

    # ax[1].contourf(tau_list, f_list, np.log10(np.abs(spectrogr)/15),v,extend="both")
    cba = ax_comp[0].contourf(tau_list, f_list, test,v,cmap=cmap,extend="both",alpha=0.05)
    # ax_comp[0].set_yticks([300,600,900,1200,1500,1800,2100,2400])
    tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
    (freq, Sigs) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')
    
    # ax_comp[1].set_xlabel("Time [s]")

    freq_reduc,Sigs_reduc=plot_interp_minimize(freq, Sigs,7000)
    time_reduc,straincheck_reduc=plot_interp_minimize(timecheck, straincheck,7000)

    
    A_square=fft(straincheck)
    freq_space = np.linspace(0.0, 1.0/(2.0*delta_t_mean), A_square.size//2)
    dE_df= 3/5 * (cn.G.cgs.value/cn.c.cgs.value**5) * (2*np.pi*freq_space)**2 * np.abs(A_square[:straincheck.size//2]*2/straincheck.size)**2
    h_char=(1/(10*cn.pc.cgs.value*1e3) * np.sqrt((2/np.pi**2) * (cn.G.cgs.value/cn.c.cgs.value**3) * dE_df))

    # freq_space_reduc,h_char_reduc=plot_interp_minimize(freq_space, h_char,1000)
    # ax_comp[2].loglog(freq_space,savgol_filter(h_char,30,2),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.5 ,lw=0.8)
    if 'SFHo' in mdl:
        ax_comp[1].loglog(freq_space[freq_space<1350],savgol_filter(h_char[freq_space<1350],45,2),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=globals()['alpha'+mdl],lw=1.)
        ax_comp[1].loglog(freq_space[freq_space>1350],savgol_filter(h_char[freq_space>1350],200,2),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=globals()['alpha'+mdl],lw=1.)
    else:
        ax_comp[1].loglog(freq_space[freq_space<1300],savgol_filter(h_char[freq_space<1300],50,2),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=globals()['alpha'+mdl],lw=1.)
        ax_comp[1].loglog(freq_space[freq_space>1300],savgol_filter(h_char[freq_space>1300],200,2),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=globals()['alpha'+mdl],lw=1.)

    # ax_comp[3].set_ylim([1e91,0.15e98])
    ax_comp[1].set_xlim([6e2,1e4])
    ax_comp[1].set_ylim(1e-22,1e-19)

    # fixed_nuevent= np.interp1d(time_reduc,time_fucking,event_fucking)
    nevent[nevent<200]=np.nan
    ax_comp[2].plot(time_nu,nevent ,color=globals()['color'+mdl],ls=globals()['ticks'+mdl],lw=1.2,alpha=globals()['alpha'+mdl] )

   
    
    time_N2,kin_erg=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["kin_erg"],100)
    time_N2,conv_max_rad=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["conv_max_rad"],100)
    time_N2,conv_min_rad=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["conv_min_rad"],100)
    time_N2,total_mass=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["total_mass"],100)
    time_shock,kinetic_shock=plot_interp(globals()[mdl+'shock'][0]-globals()['tb'+mdl], globals()[mdl+'shock'][3])
    time_N2,r_shock_in=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["r_shock_in"],100)
    time_N2,r_shock_out=plot_interp_minimize(globals()[mdl]["time"]-globals()['tb'+mdl], globals()[mdl]["r_shock_out"],100)
    time_shock,mass_shock=plot_interp(globals()[mdl+'shock'][0]-globals()['tb'+mdl], globals()[mdl+'shock'][2])
    
    total_mass[total_mass<1e33]=np.nan
    conv_min_rad[conv_min_rad<5]=np.nan
    conv_max_rad[conv_max_rad<20]=np.nan

    ax_comp[4].plot(time_N2,kin_erg/1e49,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)    
    ax_comp[3].plot(time_N2,conv_max_rad,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)        
    ax_comp[3].plot(time_N2,conv_min_rad,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)
    ax_comp[5].plot(time_N2,total_mass/2e33,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)    
    

    
    ax_comp[7].plot(time_shock,savgol_filter(kinetic_shock/1e49,3000,2),color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)    
    ax_comp[6].plot(time_N2,r_shock_out,color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)        
    ax_comp[6].plot(time_N2,savgol_filter(r_shock_in,10,2),color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)
    ax_comp[8].semilogy(time_shock,savgol_filter(mass_shock/2e33,500,2),color=globals()['color'+mdl],alpha=globals()['alpha'+mdl],ls=globals()['ticks'+mdl],lw=1.)    


    ax_comp[5].set_xlabel('t$_{pb}$ [s]',fontsize=22)

    ax_comp[3].set_xlim([0.01,1.09])
    ax_comp[4].set_xlim([0.01,1.09])
    ax_comp[5].set_xlim([0.01,1.09])
    
    ax_comp[4].set_ylim([0.01,4.1])
    ax_comp[3].set_ylim([0,100])
    ax_comp[5].set_ylim([0,3])

    
    ax_comp[6].set_xlim([0.01,1.09])
    ax_comp[7].set_xlim([0.01,1.09])
    ax_comp[8].set_xlim([0.01,1.09])
    ax_comp[6].set_ylim([0.01,250])
    ax_comp[7].set_ylim([0.01,3])
    ax_comp[8].set_ylim([0.001,5e-2])

    ax_comp[2].set_ylim([200,700])
    ax_comp[0].set_xlim([0.01,1.09])
    ax_comp[0].set_ylim([300,2400])
    ax_comp[2].set_xlim([0.01,1.09])
    number=number+1
# figure.colorbar(cba, ax=ax_comp[:], location='right',ticks=[-3,-2,-1],pad=-1.)
for num in range(3):
    ax_comp[num+6].set_xlabel('t$_{pb}$ [s]',fontsize=22)
ax_comp[1].set_title('frequency [Hz]')
ax_comp[1].tick_params(axis='x',which='major',labeltop=True,labelbottom=False)
# ax_comp[0].set_xlabel('frequency [Hz]')
# plt.minorticks_on()
ax_comp[0].set_ylabel(r"Frequency [Hz]")
ax_comp[1].set_ylabel(r"h$_{char}$")
ax_comp[2].set_ylabel(r"$\nu$ count")
ax_comp[3].set_ylabel(r'R$^{PNS}_{N^2<0}$ [km]',fontsize=22)
ax_comp[4].set_ylabel(r'E$^{PNS}_{kin,N^2<0}$' +'\n'+' [x10$^{49}$ erg]',fontsize=22)
ax_comp[5].set_ylabel(r'M$^{PNS}_{N^2<0}$ [M$_{\odot}$]',fontsize=22)

ax_comp[6].set_ylabel(r'R$^{gain}$ [km]',fontsize=22)
ax_comp[7].set_ylabel(r'E$^{gain}_{kin}$' +'\n'+' [x10$^{49}$ erg]',fontsize=22)
ax_comp[8].set_ylabel(r'M$^{gain}$ [M$_{\odot}$]',fontsize=22)

# ax_comp[0].plot(LIGO[0],LIGO[1],color='g')
# ax_comp[2].loglog(KAGRA[0],KAGRA[1],'--k')

figure.savefig('gws_SRO.pdf',bbox_inches='tight')
figure.savefig('gws_SFHo.png',bbox_inches='tight')
plt.subplots_adjust(wspace=0.4)


if  any('SRO' in name for name in reversed(sorted_names.keys())):
    black_line1, = ax_comp[3].plot([], [], color='k', linestyle='-')
    black_line2, = ax_comp[3].plot([], [], color='k', linestyle='--')
    ax_comp[3].legend([black_line1,black_line2],[r"Reference",r"$\kappa^*$"],fontsize=20,handlelength=1,frameon=False,loc= "upper right")
    
    black_line1, = plt.plot([], [], color='b', linestyle='-')
    black_line2, = plt.plot([], [], color='g', linestyle='-')
    leg=ax_comp[4].legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SRO',title_fontsize=20,frameon=False,loc= "upper right")
                     
if  any('SFHo' in name for name in reversed(sorted_names.keys())):
    black_line1, = ax_comp[3].plot([], [], color='k', linestyle='-')
    black_line2, = ax_comp[3].plot([], [], color='k', linestyle='--')
    ax_comp[3].legend([black_line1,black_line2],[r"Reference",r"$\kappa^*$"],fontsize=18,handlelength=1,frameon=False,loc= "upper right")
    
    black_line1, = plt.plot([], [], color='r', linestyle='-')
    black_line2, = plt.plot([], [], color='orange', linestyle='-')
    leg=ax_comp[2].legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SFHo',title_fontsize=20,frameon=False,loc= "lower right")
    
    black_line1, = plt.plot([], [], color='brown', linestyle=':')
    black_line2, = plt.plot([], [], color='green', linestyle=':')
    leg=ax_comp[0].legend([black_line1,black_line2],[r"SFHo",r"SRO"],fontsize=18,handlelength=1,title='Power gap',title_fontsize=20,frameon=False,loc= "lower right")
    
if  any('e8' in name for name in reversed(sorted_names.keys())):
    l1, = ax_comp[5].plot([], [], color=col[0], linestyle='-')
    l2, = ax_comp[5].plot([], [], color=col[1], linestyle='-')
    l3, = ax_comp[5].plot([], [], color=col[2], linestyle='-')
    l4, = ax_comp[5].plot([], [], color=col[3], linestyle='-')
    l5, = ax_comp[5].plot([], [], color=col[4], linestyle='-')
    l6, = ax_comp[5].plot([], [], color=col[5], linestyle='-')
    leg1 =ax_comp[5].legend([l1,l2,l3,l4,l5,l6],[r"1 $\times$ 10$^8$ GeV",r"1.5 $\times$ 10$^8$ GeV",r"2 $\times$ 10$^8$ GeV",\
                                            r"3 $\times$ 10$^8$ GeV",r"5 $\times$ 10$^8$ GeV","reference"]\
                        ,loc='lower right',handlelength=1.1,fontsize=18,title=r'f_a ',title_fontsize=20,frameon=False)
    ax_comp[5].add_artist(leg1)


# for z in range(3) : 

#     ax_comp[z].xaxis.set_minor_locator(AutoMinorLocator())
#     ax_comp[z].yaxis.set_minor_locator(AutoMinorLocator())
#     ax_comp[z].yaxis.set_ticks_position('both')
#     ax_comp[z].xaxis.set_ticks_position('both')
#     ax_comp[z].tick_params(axis='both', which='major', labelsize=18,width=2,length=4)
#     ax_comp[z].tick_params(axis='both', which='minor', labelsize=18,width=1.5,length=2)

ax_comp[0].hlines(1156,0,1.2,'green',ls=":",alpha=0.9,label='SRO')
ax_comp[0].hlines(1260,0,1.2,'brown',ls=":",alpha=0.9,label='SFHo')
ax_comp[1].vlines(1156,1e-22,1e-19,'green',ls=":",alpha=0.9,label='SRO')
ax_comp[1].vlines(1260,1e-22,1e-19,'brown',ls=":",alpha=0.9,label='SFHo')
plt.subplots_adjust(hspace=0.)


# In[15]:


# mdl = 's12'
# first_index = np.where(evan_models[mdl][:] > bt_em[mdl])[0][0]#this should be bounce time
# last_index = np.where(evan_models[mdl][first_index:] - bt_em[mdl] >HAR06K095_grey_models[mdl][-1]+bt_g10s[mdl] )[0][0]
prefactor = (3/2)*(cn.G/cn.c**4).cgs.value 
    #first_index = 0
    # Extract the 'time' column as an array
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
figure,ax_comp= plt.subplots(figsize=(10,30),nrows=6,ncols=1,sharey='row')
import seaborn as sns

def func(xy, a, b, c, d, e, f): 
    x, y = xy 
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y


def fit_line(x, y):
    # given one dimensional x and y vectors - return x and y for fitting a line on top of the regression
    # inspired by the numpy manual - https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html 
    # x = x.to_numpy() # convert into numpy arrays
    # y = y.to_numpy() # convert into numpy arrays

    A = np.vstack([x, np.ones(len(x))]).T # sent the design matrix using the intercepts
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c

dic={}
number=0
for mdl in reversed(sorted_names.keys()):
# for mdl in ['s20_ref_Gang_SFHo_new8.dat']:
    
    time_nu,nevent=nu_events[mdl[:-4]+'.txt']
    if 'hr' in mdl:
        continue
    time = evan_models[mdl][:,0]-bt_em[mdl] # this time should be bounce time
    strain = evan_models[mdl][:,59]+evan_models[mdl][:,53]+evan_models[mdl][:,47]

    timecheck1,straincheck1=plot_interp(time,strain)
    timecheck=timecheck1[timecheck1>0]
    straincheck=straincheck1[timecheck1>0]
    straincheck = np.gradient(straincheck,timecheck)
    
    # f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
    #ax.plot(time,strain)
    # ax.plot(timecheck,straincheck*prefactor)
    # ax.set_xlabel("Time post bounce")
    #ax.set_xlim([-0.01,0.6])
    # ax.set_ylim([-40,40])
    

    
    
    
    fiper = interpolate.interp1d(timecheck,straincheck*prefactor)
    tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
    (freqed, Sigsed) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')
    
    
    
    
    fiper = interpolate.interp1d(timecheck,straincheck)
    
    delta_t_mean = np.mean(np.diff(timecheck))
    sample_frequency_mean = 1/delta_t_mean
    
    f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                              hann_window=35e-3, overlap_fact=0.95, nfft_fact=1, 
                                              name=None, plot=False, energy=False,
                                             window_name='blackman')
    
    test=np.log10(np.abs(spectrogr)/15)
    test[test<-1.]=np.nan
    
    v = np.linspace(-3,0,40)

    # norm=plt.Normalize(-2,2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none",globals()['color'+mdl]])

    # ax[1].contourf(tau_list, f_list, np.log10(np.abs(spectrogr)/15),v,extend="both")
    cba = ax_comp[1].contourf(tau_list, f_list, test,v,cmap=cmap,extend="both",alpha=0.1)
    ax_comp[1].set_yticks([300,600,900,1200,1500,1800,2100,2400])



    time_reduc,strain_reduc=plot_interp_minimize(timecheck,straincheck,int(800*timecheck.max()))
    ax_comp[0].plot(time_reduc,strain_reduc*prefactor,lw=0.5,color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.2)
    # ax[0,1].plot(timecheckHA,straincheckHA*prefactor,lw=2)
   


    
    ax_comp[1].set_ylim([0,2500])
    ax_comp[2].set_ylim([0,700])
    ax_comp[1].set_xlim([0.01,1.09])
    ax_comp[0].set_xlim([0.01,1.09])
    ax_comp[2].set_xlim([0.01,1.09])
    #ax[1,1].axhline(1100,c='white',ls="--",lw=2)
    #ax[1].axhline(1100,c='white',ls="--",lw=2)
    ax_comp[1].set_yticks([300,600,900,1200,1500,1800,2100,2400])
    ax_comp[0].set_yticks([-30,-15,0,15,30])
    ax_comp[0].set_ylim([-35,35])
    # ax_comp[1].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])

    
#cb = plt.colorbar(cba,ax=ax[1,1],location='right')
    ax_comp[0].set_title(mdl+" tb="+str(expl_time[mdl])[:5],fontsize=12)
    
    tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
    (freq, Sigs) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')
    
    ax_comp[2].set_xlabel("Time [s]")

    freq_reduc,Sigs_reduc=plot_interp_minimize(freq, Sigs,7000)
    time_reduc,straincheck_reduc=plot_interp_minimize(timecheck, straincheck,7000)

    
    A_square=fft(straincheck)
    freq_space = np.linspace(0.0, 1.0/(2.0*delta_t_mean), A_square.size//2)
    dE_df= 3/5 * (cn.G.cgs.value/cn.c.cgs.value**5) * (2*np.pi*freq_space)**2 * np.abs(A_square[:straincheck.size//2]*2/straincheck.size)**2
    h_char=1/(10*cn.pc.cgs.value*1e3) * np.sqrt((2/np.pi**2) * (cn.G.cgs.value/cn.c.cgs.value**3) * dE_df)

    freq_space_reduc,h_char_reduc=plot_interp_minimize(freq_space[freq_space>1300], h_char[freq_space>1300],500)
    # ax_comp[3].loglog(freq_space,h_char,color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.2 ,lw=0.5)
    if 'SRO' in mdl:
        ax_comp[3].loglog(freq_space[freq_space<1400],savgol_filter(h_char[freq_space<1400],70,3),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.7,lw=0.7)
        ax_comp[3].loglog(freq_space[freq_space>1400],savgol_filter(h_char[freq_space>1400],300,3),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.7,lw=0.7)
    else:
        ax_comp[3].loglog(freq_space[freq_space<1300],savgol_filter(h_char[freq_space<1300],70,3),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.,lw=0.7)
        ax_comp[3].loglog(freq_space[freq_space>1300],savgol_filter(h_char[freq_space>1300],300,3),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.,lw=0.7)
    
    ax_comp[3].set_xlabel('frequency [Hz]')
    # ax_comp[3].set_ylim([1e91,0.15e98])
    # ax_comp[3].set_xlim([1e2,1e4])
    # ax_comp[3].set_ylim(1e-21,1e-19)

    # fixed_nuevent= np.interp1d(time_reduc,time_fucking,event_fucking)
    ax_comp[2].plot(time_nu,nevent ,color=globals()['color'+mdl],ls=globals()['ticks'+mdl],lw=0.7 )

    ax_comp[5].loglog(freqed, Sigsed,color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.7,lw=0.7)
    
    # Perform curve fitting 
    

    
    

    
    number=number+1
# figure.colorbar(cba, ax=ax_comp[:], location='right',ticks=[-3,-2,-1],pad=-1.)
plt.minorticks_on()
ax_comp[0].set_ylabel(r"$\mathrm{h}_x \mathrm{D}$ [cm]")
ax_comp[1].set_ylabel(r"$\mathrm{Frequency}$ [Hz] ")
ax_comp[3].set_ylabel(r"h$_{char}")
ax_comp[2].set_ylabel(r"$\nu$ count")

figure.savefig('gws_SRO.pdf',bbox_inches='tight')
figure.savefig('gws_SFHo.png',bbox_inches='tight')
plt.subplots_adjust(wspace=0)



# In[24]:


# mdl='s20_simp_SRO3.dat'
import cv2
for mdl in reversed(sorted_names.keys()):
    fig,ax=plt.subplots(1,1,figsize=(15,15))
    time = evan_models[mdl][:,0]-bt_em[mdl] # this time should be bounce time
    strain = evan_models[mdl][:,59]+evan_models[mdl][:,53]+evan_models[mdl][:,47]
    
    timecheck1,straincheck1=plot_interp(time,strain)
    timecheck=timecheck1[timecheck1>0]
    straincheck=straincheck1[timecheck1>0]
    straincheck = np.gradient(straincheck,timecheck)
    # f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
    #ax.plot(time,strain)
    # ax.plot(timecheck,straincheck*prefactor)
    # ax.set_xlabel("Time post bounce")
    #ax.set_xlim([-0.01,0.6])
    # ax.set_ylim([-40,40])
    
    fiper = interpolate.interp1d(timecheck,straincheck)
    
    delta_t_mean = np.mean(np.diff(timecheck))
    sample_frequency_mean = 1/delta_t_mean
    
    f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                              hann_window=35e-3, overlap_fact=0.95, nfft_fact=1, 
                                              name=None, plot=False, energy=False,
                                             window_name='blackman')
    
    
    test=np.log10(np.abs(spectrogr)/15)
    test[test<-1.8]=np.nan
    
    v = np.linspace(-3,0,40)
    
    # norm=plt.Normalize(-2,2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none",globals()['color'+mdl]])
    
    ax.contourf(tau_list, f_list, test,v,extend="both",cmap ='Greys')
    
    ax.set_ylim([0,2500])
    ax.set_xlim([0.01,1.09])
    plt.subplots_adjust(wspace=0,hspace=0)
    ax.set_xticklabels([])    
    ax.set_yticklabels([])    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(mdl[:-4]+'src.png')


# In[ ]:


# mdl='s20_simp_SRO3.dat'
import cv2
fig,ax=plt.subplots(1,1,figsize=(15,15))

for mdl in reversed(sorted_names.keys()):
    time = evan_models[mdl][:,0]-bt_em[mdl] # this time should be bounce time
    strain = evan_models[mdl][:,59]+evan_models[mdl][:,53]+evan_models[mdl][:,47]
    
    timecheck1,straincheck1=plot_interp(time,strain)
    timecheck=timecheck1[timecheck1>0]
    straincheck=straincheck1[timecheck1>0]
    straincheck = np.gradient(straincheck,timecheck)
    # f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
    #ax.plot(time,strain)
    # ax.plot(timecheck,straincheck*prefactor)
    # ax.set_xlabel("Time post bounce")
    #ax.set_xlim([-0.01,0.6])
    # ax.set_ylim([-40,40])
    
    fiper = interpolate.interp1d(timecheck,straincheck)
    
    delta_t_mean = np.mean(np.diff(timecheck))
    sample_frequency_mean = 1/delta_t_mean
    
    f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                              hann_window=35e-3, overlap_fact=0.95, nfft_fact=1, 
                                              name=None, plot=False, energy=False,
                                             window_name='blackman')
    
    
    test=np.log10(np.abs(spectrogr)/15)
    test[test>-1.]=np.nan
    
    v = np.linspace(-3,0,40)
    
    # norm=plt.Normalize(-2,2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none",globals()['color'+mdl]])
    
    ax.contourf(tau_list, f_list, test,v,extend="both",cmap ='Greys_r',alpha=0.05)
    
    ax.set_ylim([0,2500])
    ax.set_xlim([0.01,1.09])
    plt.subplots_adjust(wspace=0,hspace=0)
    ax.set_xticklabels([])    
    ax.set_yticklabels([])    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    # plt.savefig(mdl[:-4]+'src.png')


# In[27]:


# plt.figure(1)    
dic={}
fig,ax=plt.subplots(1,1,figsize=(15,15))

for mdl in reversed(sorted_names.keys()): 
# for mdl in ['s20_1_5e8.dat']:
    # if "SRO" in mdl: 
        # continue
    # elif "ax_on" in mdl or 'e8' in mdl : 
        # continue
    # else:
    
    time = evan_models[mdl][:,0]-bt_em[mdl] # this time should be bounce time
    strain = evan_models[mdl][:,59]+evan_models[mdl][:,53]+evan_models[mdl][:,47]
    
    timecheck1,straincheck1=plot_interp(time,strain)
    timecheck=timecheck1[timecheck1>0]
    straincheck=straincheck1[timecheck1>0]
    straincheck = np.gradient(straincheck,timecheck)
    # f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
    #ax.plot(time,strain)
    # ax.plot(timecheck,straincheck*prefactor)
    # ax.set_xlabel("Time post bounce")
    #ax.set_xlim([-0.01,0.6])
    # ax.set_ylim([-40,40])
    
    fiper = interpolate.interp1d(timecheck,straincheck)
    
    delta_t_mean = np.mean(np.diff(timecheck))
    sample_frequency_mean = 1/delta_t_mean
    
    f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                              hann_window=35e-3, overlap_fact=0.95, nfft_fact=1, 
                                              name=None, plot=False, energy=False,
                                             window_name='blackman')
    
    
    test=np.log10(np.abs(spectrogr)/15)
    test[test<-1.8]=np.nan
    
    img = cv2.imread(mdl[:-4]+'src.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    
    low_threshold = 50
    high_threshold = 150
    
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 00  # minimum number of pixels making up a line
    max_line_gap = 200  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    xes=[]
    yes=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            xes.append(x1)
            xes.append(x2)
            yes.append(y1)
            yes.append(y2)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    # plt.imshow(lines_edges)
    # tau_new= np.interp(time,sorted_x,sorted_y)

    
    test1,test2=plot_interp(xes,yes)
    # f = scipy.interpolate.interp1d(test2,)
    # time = np.linspace(0,1.1,len(test2))
    # print((test2.min()))
    
    # tau=np.interp(time,test1,test2)
    # test3,test4=plot_interp_minimize(x,test2,500)
    time_shaped=((test1[::-1]-test1.min())*( tau_list.max()/(test1.max()-test1.min())))
    
    freq_shaped=(-test2[::-1]+test2.max())*(2400/test2.max())
    
    # freq_shaped[freq_shaped>1900]=np.nan
    fit= np.polyfit(time_shaped, freq_shaped, 2)
    p = np.poly1d(fit)
    ax.plot(time_shaped,p(time_shaped),color=globals()['color'+mdl],ls=globals()['ticks'+mdl],alpha=0.7,lw=0.7)
   
    dic[mdl]=[time_shaped,p(time_shaped)]
    
    v = np.linspace(-3,0,40)
    
    # norm=plt.Normalize(-2,2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none",globals()['color'+mdl]])
    
    ax.contourf(tau_list, f_list, test,v,extend="both",cmap =cmap,alpha=0.1)
    print(mdl,time[-1])
np.save('spectro_fit.npy',dic)
ax.set_xlim([0,1.1])
ax.set_ylim([0,2400])
ax.hlines(1230,0,1.2,'k',alpha=0.7)
ax.hlines(1200,0,1.2,'b',alpha=0.7)


# In[ ]:


mdl = 's12'
first_index = np.where(evan_models[mdl][:,0] > bt_em[mdl])[0][0]#this should be bounce time
last_index = np.where(evan_models[mdl][first_index:,0] - bt_em[mdl] >HAR06K095_grey_models[mdl][-1,0]+bt_g10s[mdl] )[0][0]
prefactor = (3/2)*(cn.G/cn.c**4).cgs.value 
#first_index = 0
# Extract the 'time' column as an array
timecheck = evan_models[mdl][first_index:last_index,0]-bt_em[mdl] # this time should be bounce time
straincheck = evan_models[mdl][first_index:last_index,59]+evan_models[mdl][first_index:last_index,53]+evan_models[mdl][first_index:last_index,47]
straincheck = np.gradient(straincheck,timecheck)

f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
#ax.plot(time,strain)
ax.plot(timecheck,straincheck*prefactor)
ax.set_xlabel("Time post bounce")
#ax.set_xlim([-0.01,0.6])
ax.set_ylim([-40,40])



fiper = interpolate.interp1d(timecheck,straincheck*prefactor)
tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
(freqed, Sigsed) = scipy.signal.periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')




fiper = interpolate.interp1d(timecheck,straincheck)

delta_t_mean = np.mean(np.diff(timecheck))
sample_frequency_mean = 1/delta_t_mean

f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                          hann_window=40e-3, overlap_fact=0.99, nfft_fact=1, 
                                          name=None, plot=False, energy=False,
                                         window_name='blackman')


timecheckHA = HAR06K095_grey_models[mdl][:,0]+bt_g10s[mdl] # this time should be bounce time
straincheckHA = HAR06K095_grey_models[mdl][:,59]+HAR06K095_grey_models[mdl][:,53]+HAR06K095_grey_models[mdl][:,47]
straincheckHA = np.gradient(straincheckHA,timecheckHA)

f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')
#ax.plot(time,strain)
ax.plot(timecheckHA,straincheckHA*prefactor)
ax.set_xlabel("Time post bounce")
#ax.set_xlim([-0.01,0.6])
ax.set_ylim([-40,40])

fiper = interpolate.interp1d(timecheckHA,straincheckHA)

delta_t_mean = np.mean(np.diff(timecheckHA))
sample_frequency_mean = 1/delta_t_mean

f_listHA, tau_listHA, spectrogrHA = spectrogram(fiper, timecheckHA, delta_t_mean, sample_frequency_mean,
                                          hann_window=40e-3, overlap_fact=0.99, nfft_fact=1, 
                                          name=None, plot=False, energy=False,
                                         window_name='blackman')

fiperHA = interpolate.interp1d(timecheckHA,straincheckHA*prefactor)
tipHA = np.linspace(timecheckHA[0],timecheckHA[-1],len(timecheckHA))
(freqg, Sigsg) = scipy.signal.periodogram(fiperHA(tipHA),1/(tipHA[1]-tipHA[0]), scaling='density')

f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')

ax.plot(freqed, Sigsed)
ax.plot(freqg, Sigsg)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
ax.set_xlim([0,2000])
ax.set_ylim([0,0.2])


# In[ ]:


from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

f,ax = plt.subplots(figsize=(16,10),nrows=2,ncols=2,sharey='row',sharex='all')
ax[0,0].plot(timecheck,straincheck*prefactor,lw=2)
ax[0,1].plot(timecheckHA,straincheckHA*prefactor,lw=2)

v = np.linspace(-3,0,40)
ax[1,0].contourf(tau_list, f_list, np.log10(np.abs(spectrogr)/15),v,extend="both")
cba = ax[1,1].contourf(tau_listHA, f_listHA, np.log10(np.abs(spectrogrHA)/15),v,extend="both")
ax[1,1].set_ylim([0,2500])
ax[1,1].set_xlim([0,0.6])
#ax[1,1].axhline(1100,c='white',ls="--",lw=2)
#ax[1,0].axhline(1100,c='white',ls="--",lw=2)
ax[1,0].set_yticks([300,600,900,1200,1500,1800,2100,2400])
ax[0,0].set_yticks([-30,-15,0,15,30])
ax[0,0].set_ylim([-35,35])
ax[1,0].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])
plt.subplots_adjust(wspace=0,hspace=0)
ax[1,0].set_xlabel("Time [s]")
ax[1,1].set_xlabel("Time [s]")
ax[0,0].set_ylabel(r"$\mathrm{h}_x \mathrm{D}$ [cm]")
ax[1,0].set_ylabel(r"$\mathrm{Frequency}$ [Hz] ")
#cb = plt.colorbar(cba,ax=ax[1,1],location='right')
ax[0,0].set_title("Energy Dependent")
ax[0,1].set_title("Grey")
f.colorbar(cba, ax=ax[:,1], location='right',ticks=[-3,-2,-1,0],pad=0.01)
plt.minorticks_on()

plt.savefig('gws_s12.pdf',bbox_inches='tight')
plt.savefig('gws_s12.png',bbox_inches='tight')


# In[ ]:


#import seaborn as sns
#colors = sns.color_palette()
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from scipy.signal import savgol_filter

modl='s12'
f,ax = plt.subplots(figsize=(7,12),nrows=2,ncols=1,sharey='row',sharex='all')


ax[0].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,33], window_length=150, polyorder=3),
             lw=3,color='C0',label=r"$\nu_e$")
ax[0].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,33], window_length=150, polyorder=3)
             ,lw=3,color='C0',ls='--')


ax[0].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,34], window_length=150, polyorder=3),
             lw=3,color='C2',label=r"$\bar{\nu}_e$")
ax[0].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,34], window_length=150, polyorder=3)
             ,lw=3,color='C2',ls='--')

ax[0].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,35], window_length=150, polyorder=3),
             lw=3,color='C3',label=r"$\nu_x$")
ax[0].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,35], window_length=150, polyorder=3)
             ,lw=3,color='C3',ls='--')

ax[0].set_ylim([0,120])
ax[0].set_xlim([0.01,0.6])

ax[1].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,36], window_length=150, polyorder=3),
             lw=3,color='C0')
ax[1].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,36], window_length=150, polyorder=3)
             ,lw=3,color='C0',ls='--')


ax[1].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,37], window_length=150, polyorder=3),
             lw=3,color='C2')
ax[1].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,37], window_length=150, polyorder=3)
             ,lw=3,color='C2',ls='--')

ax[1].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,38], window_length=150, polyorder=3),
             lw=3,color='C3')
ax[1].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,38], window_length=150, polyorder=3)
             ,lw=3,color='C3',ls='--')

ax[1].set_ylim([5,32])
ax[1].set_xlim([0.01,0.6])
ax[0].legend(loc=0,frameon=False)
#ax[1,1].axhline(1100,c='white',ls="--",lw=2)
#ax[1,0].axhline(1100,c='white',ls="--",lw=2)
#ax[1,0].set_yticks([300,600,900,1200,1500,1800,2100,2400])
#ax[0,0].set_yticks([-30,-15,0,15,30])
#ax[0,0].set_ylim([-35,35])
#ax[1,0].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])
plt.subplots_adjust(wspace=0,hspace=0)
ax[1].set_xlabel("Time [s]")
ax[1].set_xlabel("Time [s]")
ax[0].set_ylabel(r"$\mathrm{Luminosity}$ [$10^{51}$ erg/s ]")
ax[1].set_ylabel(r"$\mathrm{Average \ Energy}$ [MeV]")
#cb = plt.colorbar(cba,ax=ax[1,1],location='right')
#ax[0].set_title("Energy Dependent")

#f.colorbar(cba, ax=ax[:,1], location='right',ticks=[-3,-2,-1,0],pad=0.01)
plt.minorticks_on()

plt.savefig('lums12.pdf',bbox_inches='tight')
plt.savefig('lums12.png',bbox_inches='tight')

#HAR06K095_grey_models


# In[ ]:


#import seaborn as sns
#colors = sns.color_palette()
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from scipy.signal import savgol_filter


modl='s14'
f,ax = plt.subplots(figsize=(10,17),nrows=2,ncols=1,sharey='row',sharex='all')


ax[0].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,33], window_length=100, polyorder=3),
             lw=3,color='C0',label=r"$\nu_e$")
ax[0].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,33], window_length=100, polyorder=3)
             ,lw=3,color='C0',ls='--')


ax[0].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,34], window_length=100, polyorder=3),
             lw=3,color='C2',label=r"$\bar{\nu}_e$")
ax[0].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,34], window_length=100, polyorder=3)
             ,lw=3,color='C2',ls='--')

ax[0].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,35], window_length=100, polyorder=3),
             lw=3,color='C3',label=r"$\nu_x$")
ax[0].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,35], window_length=100, polyorder=3)
             ,lw=3,color='C3',ls='--')

ax[0].set_ylim([10,127])
ax[0].set_xlim([0.01,0.6])

ax[1].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,36], window_length=100, polyorder=3),
             lw=3,color='C0')
ax[1].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,36], window_length=100, polyorder=3)
             ,lw=3,color='C0',ls='--')


ax[1].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,37], window_length=100, polyorder=3),
             lw=3,color='C2')
ax[1].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,37], window_length=100, polyorder=3)
             ,lw=3,color='C2',ls='--')

ax[1].plot(evan_models[modl][:,0]-bt_em[modl],
             savgol_filter(evan_models[modl][:,38], window_length=100, polyorder=3),
             lw=3,color='C3')
ax[1].plot(HAR06K095_grey_models[modl][:,0]+bt_g10s[modl],
             savgol_filter(HAR06K095_grey_models[modl][:,38], window_length=100, polyorder=3)
             ,lw=3,color='C3',ls='--')

ax[1].set_ylim([5,32])
ax[1].set_xlim([0.01,0.6])
ax[0].legend(loc=0,frameon=False)
#ax[1,1].axhline(1100,c='white',ls="--",lw=2)
#ax[1,0].axhline(1100,c='white',ls="--",lw=2)
#ax[1,0].set_yticks([300,600,900,1200,1500,1800,2100,2400])
#ax[0,0].set_yticks([-30,-15,0,15,30])
#ax[0,0].set_ylim([-35,35])
#ax[1,0].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])
plt.subplots_adjust(wspace=0,hspace=0)
ax[1].set_xlabel("Time [s]")
ax[1].set_xlabel("Time [s]")
ax[0].set_ylabel(r"$\mathrm{Luminosity}$ [$10^{51}$ erg/s ]")
ax[1].set_ylabel(r"$\mathrm{Average \ Energy}$ [MeV]")
#cb = plt.colorbar(cba,ax=ax[1,1],location='right')
#ax[0].set_title("Energy Dependent")

#f.colorbar(cba, ax=ax[:,1], location='right',ticks=[-3,-2,-1,0],pad=0.01)
plt.minorticks_on()

plt.savefig('lums12.pdf',bbox_inches='tight')
plt.savefig('lums12.png',bbox_inches='tight')

#HAR06K095_grey_models


# In[ ]:


def high_pass_filter(signal, cutoff_frequency, fs):
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(N=5, Wn=normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


# In[ ]:


modl='s12'

fiperHA = interpolate.interp1d(HAR06K095_grey_models[modl][:,0],HAR06K095_grey_models[modl][:,33])
tipHA = np.linspace(HAR06K095_grey_models[modl][0,0],
                    HAR06K095_grey_models[modl][-1,0],
                    len(HAR06K095_grey_models[modl][:,0]))

dt = tipHA[1]-tipHA[0]
ds = 1/dt
signal = fiperHA(tipHA)
signal = high_pass_filter(signal,10.0,ds)
(freqg, Sigsg) = scipy.signal.periodogram(signal,ds, scaling='density')

f,ax = plt.subplots(figsize=(8,4),nrows=1,ncols=1,sharey='col',sharex='all')

#ax.plot(freqed, Sigsed)
ax.plot(freqg, Sigsg)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
ax.set_xlim([0,500])
ax.set_ylim([0,.5])


# In[ ]:




