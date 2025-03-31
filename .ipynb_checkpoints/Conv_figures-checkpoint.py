import matplotlib.pylab as plt
import numpy as np
import Shared

def Plot_convection_zone(time_range):
    fig,ax=plt.subplots(3,1,figsize=(8,10))
    
    for base in  Shared.all_simulations.keys():

        #### Read Conv files
        
        Shared.all_simulations[base]["Conv"]={}
        try:
            Shared.all_simulations[base]["Conv"]=np.load('N2_values/allprofiles'+base[:-4]+'.npy',allow_pickle=True).item()
        except: 
            print("nope, no conv found",base)
            continue

        
        ##############  clean stuff to not have weird drops 
        # Shared.all_simulations[base]["Conv"]["time"][Shared.all_simulations[base]["time"]==0.]=np.nan
        # Shared.all_simulations[base]["Conv"]["conv_min_rad"][Shared.all_simulations[base]["conv_min_rad"]<5]=np.nan
        # Shared.all_simulations[base]["Conv"]["conv_max_rad"][Shared.all_simulations[base]["conv_max_rad"]==0.]=np.nan
        # Shared.all_simulations[base]["Conv"]["kin_erg"][Shared.all_simulations[base]["kin_erg"]==0.]=np.nan
        # Shared.all_simulations[base]["Conv"]["total_mass"][Shared.all_simulations[base]["total_mass"]/2e33>0.78]=np.nan
    
        # ############## plot values

        ## Radius
        time,y=Shared.plot_interp(Shared.all_simulations[base]["Conv"]['time'],Shared.all_simulations[base]["Conv"]['conv_max_rad'])
        
        ax[0].plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])

        time,y=Shared.plot_interp(Shared.all_simulations[base]["Conv"]['time'],Shared.all_simulations[base]["Conv"]['conv_min_rad'])
        
        ax[0].plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])


        ##  Kinetic energy
        time,y=Shared.plot_interp(Shared.all_simulations[base]["Conv"]['time'],Shared.all_simulations[base]["Conv"]['kin_erg'])
     
        ax[1].plot(time-Shared.all_simulations[base]['t_bounce'],y/1e49,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])
    

        ## total mass
        time,y=Shared.plot_interp(Shared.all_simulations[base]["Conv"]['time'],Shared.all_simulations[base]["Conv"]['total_mass'])
        
        ax[2].plot(time-Shared.all_simulations[base]['t_bounce'],y/2e33,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])





    ###########   labels
    ax[0].set_ylabel('R$_{N^2<0}$ [km]',fontsize=20)
    ax[1].set_ylabel(r'$E_{kin,N^2<0}$' +'\n'+' [x10$^{49}$ erg]',fontsize=19)
    ax[2].set_xlabel('t$_{pb}$ [s]',fontsize=19)
    ax[2].set_ylabel('M$_{PNS,N^2<0}$ [M$_{\odot}$]',fontsize=19)

    ######### ranges 
    ax[0].set_xlim(time_range)
    ax[1].set_xlim(time_range)
    ax[2].set_xlim(time_range)
    ax[0].set_ylim([0.01,55])

    ax[1].set_ylim([0.01,1.49])
    ax[0].set_ylim([0,100])

    ################## legends 
    col = plt.cm.jet(np.linspace(1,0,7))
    if  any('SRO' in name for name in  Shared.all_simulations.keys()):
        black_line1, = ax[0].plot([], [], color='k', linestyle='-')
        black_line2, = ax[0].plot([], [], color='k', linestyle='--')
        ax[0].legend([black_line1,black_line2],[r"Reference",r"$\kappa^*$"],fontsize=22,handlelength=1,frameon=False,loc= "upper right")
        
        black_line1, = plt.plot([], [], color='b', linestyle='-')
        black_line2, = plt.plot([], [], color='g', linestyle='-')
        leg=ax[1].legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=22,handlelength=1,title_fontsize=22,frameon=False,loc= "upper right")
                         
    if  any('SFHo' in name for name in  Shared.all_simulations.keys()):
        black_line1, = ax[0].plot([], [], color='k', linestyle='-')
        black_line2, = ax[0].plot([], [], color='k', linestyle='--')
        ax[0].legend([black_line1,black_line2],[r"Reference",r"$\kappa^*$"],fontsize=22,handlelength=1,frameon=False,loc= "upper right")
        
        black_line1, = plt.plot([], [], color='r', linestyle='-')
        black_line2, = plt.plot([], [], color='orange', linestyle='-')
        leg=ax[2].legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=22,handlelength=1,title_fontsize=22,frameon=False,loc= "lower right")
    
    if  any((('ax_on' in name) or ( "e8" in name)) for name in  Shared.all_simulations.keys()):
        
        l1, = ax[0].plot([], [], color=col[0], linestyle='-')
        l2, = ax[0].plot([], [], color=col[1], linestyle='-')
        l3, = ax[0].plot([], [], color=col[2], linestyle='-')
        l4, = ax[0].plot([], [], color=col[3], linestyle='-')
        l5, = ax[0].plot([], [], color=col[4], linestyle='-')
        l6, = ax[0].plot([], [], color=col[5], linestyle='-')
        leg1 =ax[0].legend([l1,l2,l3,l4,l5,l6],[r"1 $\times$ 10$^8$ GeV",r"1.5 $\times$ 10$^8$ GeV",r"2 $\times$ 10$^8$ GeV",\
                                                r"3 $\times$ 10$^8$ GeV",r"5 $\times$ 10$^8$ GeV","reference"]\
                            ,loc='upper right',handlelength=1.1,fontsize=18,frameon=False)
        ax[0].add_artist(leg1)



    
    # ############# make pretty   
    # for z in range(3) : 
    #     ax[z].xaxis.set_minor_locator(AutoMinorLocator())
    #     ax[z].yaxis.set_minor_locator(AutoMinorLocator())
    
    #     ax[z].yaxis.set_ticks_position('both')
    #     ax[z].xaxis.set_ticks_position('both')
    #     # ax[z].grid()
    
    #     ax[z].tick_params(axis='both', which='major', labelsize=18,width=2,length=4)
    #     ax[z].tick_params(axis='both', which='minor', labelsize=18,width=1.5,length=2)
    #     plt.subplots_adjust(wspace=0,hspace=0.)
    ax[0].set_xticklabels([])    
    ax[1].set_xticklabels([]) 


    
    plt.savefig('figures/Convection.png')