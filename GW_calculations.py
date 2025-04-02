# base='s20_simp_SRO3.dat'
import cv2
import Shared
import matplotlib

import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
from scipy.signal import periodogram,savgol_filter
import scipy.signal as signal
from scipy.fftpack import fft
import astropy.constants as cn
import astropy.units as u
prefactor = (3/2)*(cn.G/cn.c**4).cgs.value 
import os

print(prefactor)
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
        
        f_list, tau_list, Zxx = signal.stft(prefactor*I2_eq,df, 
                                     nperseg=Nperseg, noverlap=Noverlap, window=window_name, nfft=Nfft)
        spectrogr = np.abs(Zxx)**2
        
    if energy:
        Nperseg = int(hann_window/delta_t_mean)
        Nfft = Nperseg*nfft_fact
        #print(len(time_list)/Nperseg)
        f_list, tau_list, Zxx = signal.stft(I2_interp(time_list), sample_frequency_mean, 
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
        # print(np.max(np.abs(spectrogr)**2))
        plt.colorbar(orientation='horizontal')
        plt.ylim(0,2000)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time post bounce [s]')
        #plt.xlim(0,0.4)
        plt.savefig("figures/Spectrogram_"+name+'.png')
    return f_list, tau_list, spectrogr


def Image_for_linear_fit(base,time_range,f_list, tau_list, spectrogr):
    fig,ax=plt.subplots(1,1,figsize=(15,15))

    test=np.log10(np.abs(spectrogr)/15)
    test[test<-1.8]=np.nan
    
    v = np.linspace(-3,0,40)
    
    # norm=plt.Normalize(-2,2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none",Shared.all_simulations[base]['color']])
    
    ax.contourf(tau_list, f_list, test,v,extend="both",cmap ='Greys')
    
    ax.set_ylim([0,2500])
    ax.set_xlim(time_range)
    plt.subplots_adjust(wspace=0,hspace=0)
    ax.set_xticklabels([])    
    ax.set_yticklabels([])    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(base[:-4]+'src.png')




# plt.figure(1) 
def Create_linear_fit_for_fmode(base,time_range,f_list, tau_list, spectrogr):


    Image_for_linear_fit(base,time_range,f_list, tau_list, spectrogr)
    
    img = cv2.imread(base[:-4]+'src.png')
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

    
    test1,test2=Shared.plot_interp(xes,yes)

    time_shaped=((test1[::-1]-test1.min())*( tau_list.max()/(test1.max()-test1.min())))
    
    freq_shaped=(-test2[::-1]+test2.max())*(2400/test2.max())
    
    fit= np.polyfit(time_shaped, freq_shaped, 2)
    p = np.poly1d(fit)
   
    os.remove(base[:-4]+'src.png')
    # print([time_shaped,p(time_shaped)])
    return [time_shaped,p(time_shaped)]
            
    


def GWs(time_range):
    figure,ax_comp= plt.subplots(figsize=(10,30),nrows=4,ncols=1,sharey='row')

    for base in Shared.all_simulations.keys():
        if '1D' in base:
            continue
        time=Shared.all_simulations[base]['time']-Shared.all_simulations[base]['t_bounce']
        strain=Shared.all_simulations[base]['strain']
        
        timecheck1,straincheck1=Shared.plot_interp(time,strain)
        timecheck=timecheck1[timecheck1>0]
        straincheck=straincheck1[timecheck1>0]
        straincheck = np.gradient(straincheck,timecheck)

        
        
       
        fiper = interpolate.interp1d(timecheck,straincheck)
        
        delta_t_mean = np.mean(np.diff(timecheck))
        sample_frequency_mean = 1/delta_t_mean
        
        f_list, tau_list, spectrogr = spectrogram(fiper, timecheck, delta_t_mean, sample_frequency_mean,
                                                  hann_window=35e-3, overlap_fact=0.95, nfft_fact=1, 
                                                  name=base, plot=True, energy=False,
                                                 window_name='blackman')
        
        test=np.log10(np.abs(spectrogr)/15)
        test[test<-1.]=np.nan
        
        v = np.linspace(-3,0,40)
    
        # norm=plt.Normalize(-2,2)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["none",Shared.all_simulations[base]['color']])
    
        cba = ax_comp[1].contourf(tau_list, f_list, test,v,cmap=cmap,extend="both",alpha=0.1)
        ax_comp[1].set_yticks([300,600,900,1200,1500,1800,2100,2400])
        try :
            fit_time,fit_line=Create_linear_fit_for_fmode(base,time_range,f_list, tau_list, spectrogr)
            ax_comp[1].plot(fit_time,fit_line,color=Shared.all_simulations[base]['color'],ls=Shared.all_simulations[base]['ticks'],alpha=0.3)

        except: 
            print("GWs fit bugged, still making figures")
        
        

    
        time_reduc,strain_reduc=Shared.plot_interp_minimize(timecheck,straincheck,int(800*timecheck.max()))
        ax_comp[0].plot(time_reduc,strain_reduc*prefactor,lw=0.5,color=Shared.all_simulations[base]['color'],ls=Shared.all_simulations[base]['ticks'],alpha=0.2)
        fiper = interpolate.interp1d(timecheck,straincheck*prefactor)
        tip = np.linspace(timecheck[0],timecheck[-1],len(timecheck))
        (freq, Sigs) = periodogram(fiper(tip),1/(tip[1]-tip[0]), scaling='density')
        
       
    
        freq_reduc,Sigs_reduc=Shared.plot_interp_minimize(freq, Sigs,7000)
        time_reduc,straincheck_reduc=Shared.plot_interp_minimize(timecheck, straincheck,7000)
    
        
        A_square=fft(straincheck)
        freq_space = np.linspace(0.0, 1.0/(2.0*delta_t_mean), A_square.size//2)
        dE_df= 3/5 * (cn.G.cgs.value/cn.c.cgs.value**5) * (2*np.pi*freq_space)**2 * np.abs(A_square[:straincheck.size//2]*2/straincheck.size)**2
        h_char=1/(10*cn.pc.cgs.value*1e3) * np.sqrt((2/np.pi**2) * (cn.G.cgs.value/cn.c.cgs.value**3) * dE_df)
    
        freq_space_reduc,h_char_reduc=Shared.plot_interp_minimize(freq_space[freq_space>1300], h_char[freq_space>1300],500)
        # ax_comp[3].loglog(freq_space,h_char,color=Shared.all_simulations[base]['color'],ls=Shared.all_simulations[base]['ticks'],alpha=0.2 ,lw=0.5)
      
        ax_comp[2].loglog(freq_space[freq_space<1400],savgol_filter(h_char[freq_space<1400],70,3),color=Shared.all_simulations[base]['color'],ls=Shared.all_simulations[base]['ticks'],alpha=0.7,lw=0.7)
        
        ax_comp[2].loglog(freq_space[freq_space>1400],savgol_filter(h_char[freq_space>1400],300,3),color=Shared.all_simulations[base]['color'],ls=Shared.all_simulations[base]['ticks'],alpha=0.7,lw=0.7)
        
        ax_comp[3].loglog(freq_reduc,Sigs_reduc,color=Shared.all_simulations[base]['color'],ls=Shared.all_simulations[base]['ticks'],alpha=0.7,lw=0.7)


        ###################### labels & limits
        
        ax_comp[1].set_xlabel("Time [s]")
        ax_comp[3].set_xlabel('frequency [Hz]')
        ax_comp[1].set_xlim(time_range)
        ax_comp[0].set_xlim(time_range)
        ax_comp[2].set_xlim([5e2,1e6])
        ax_comp[3].set_xlim([5e2,1e6])
        
        ax_comp[1].set_ylim([0,2500])
        ax_comp[1].set_yticks([300,600,900,1200,1500,1800,2100,2400])
        ax_comp[0].set_yticks([-30,-15,0,15,30])
        ax_comp[0].set_ylim([-35,35])
        ax_comp[3].set_ylim([1e2,1e-13])


    # figure.colorbar(cba, ax=ax_comp[:], location='right',ticks=[-3,-2,-1],pad=-1.)
    plt.minorticks_on()
    ax_comp[0].set_ylabel(r"$\mathrm{h}_x \mathrm{D}$ [cm]")
    ax_comp[1].set_ylabel(r"$\mathrm{Frequency}$ [Hz] ")
    ax_comp[2].set_ylabel(r"h$_{char}$")
    
    
    figure.savefig('figures/gws_cumulated.png',bbox_inches='tight')
    plt.subplots_adjust(wspace=0)
    

