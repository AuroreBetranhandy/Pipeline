import Shared
import matplotlib.pylab as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import warnings 
warnings.simplefilter('ignore')

                               
def Plot_classic_Hydro(time_range):

        #########################  Paper ready Hydro
        factor=1e-5

        fig,ax = plt.subplots(2,1,figsize=(8,10))
        ax1=ax[0]
        ax2=ax[1]


        for base in Shared.all_simulations.keys():
            # print(base,Shared.all_simulations[base]['t_bounce'])
            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['shock_radius'])
            ax1.plot(time-Shared.all_simulations[base]['t_bounce'],y*factor,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])

            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['PNS_radius'])
            ax2.plot(time-Shared.all_simulations[base]['t_bounce'],y*factor,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])

        ############ limits

        ax1.set_xlim(time_range)
        ax2.set_xlim(time_range)

        ax1.set_ylim([41,250])
        ax2.set_ylim([30,80])

        ########### legend

        black_line1, = plt.plot([], [], color='k', linestyle='-')
        black_line2, = plt.plot([], [], color='k', linestyle='--')
        black_line3, = plt.plot([], [], color='k', linestyle=':')
        ax2.legend([black_line1,black_line2,black_line3],[r"Reference",r"$\kappa_a^*$","Full"],fontsize=18,handlelength=1,frameon=False,loc= "upper right")


        black_line1, = plt.plot([], [], color='r', linestyle='-')
        black_line2, = plt.plot([], [], color='orange', linestyle='-')
        ax1.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SFHo',title_fontsize=18,frameon=False,loc= "upper right")
          
        ax3=ax1.twinx()
        black_line1, = plt.plot([], [], color='b', linestyle='-')
        black_line2, = plt.plot([], [], color='g', linestyle='-')
        ax3.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SRO',title_fontsize=18,frameon=False,loc= "upper center")


        ########## labels
        
        ax1.set_xlabel(r't [s]',fontsize=18)
        ax2.set_xlabel(r't [s]',fontsize=18)
        ax1.set_ylabel(r'Shock radius [km]',fontsize=18)
        ax2.set_ylabel(r'PNS radius [km]',fontsize=18)
        
        ax3.set_xticklabels([])    
        ax3.set_yticklabels([])    
        
        for i in range(len(ax)-1):
            ax[i].set_xticklabels([])
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.savefig('Hydro.png')

    

def   Plot_nu_luminosity(time_range):    #########################  Paper ready Nu lum

        fig,ax = plt.subplots(3,1,figsize=(8,15))
        ax1=ax[0]
        ax2=ax[1]
        ax3=ax[2]


        for base in Shared.all_simulations.keys():
            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['nue_lumi'])
            ax1.plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])

            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['nua_lumi'])
            ax2.plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])
            
            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['nux_lumi'])
            ax3.plot(time-Shared.all_simulations[base]['t_bounce'],y/4,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])

        ############ limits


        ax1.set_xlim(time_range)
        ax2.set_xlim(time_range)
        ax3.set_xlim(time_range)

        ax1.set_ylim([10.1,100])
        ax2.set_ylim([10.1,50])
        ax3.set_ylim([10.1,40])

        ############### legend

        black_line1, = plt.plot([], [], color='k', linestyle='-')
        black_line2, = plt.plot([], [], color='k', linestyle='--')
        black_line3, = plt.plot([], [], color='k', linestyle=':')
        ax2.legend([black_line1,black_line2,black_line3],[r"Reference",r"$\kappa_a^*$","Full"],fontsize=18,handlelength=1,frameon=False,loc= "upper right")


        black_line1, = plt.plot([], [], color='r', linestyle='-')
        black_line2, = plt.plot([], [], color='orange', linestyle='-')
        ax1.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SFHo',title_fontsize=18,frameon=False,loc= "upper right")
          
        # ~ ax3=ax1.twinx()
        black_line1, = plt.plot([], [], color='b', linestyle='-')
        black_line2, = plt.plot([], [], color='g', linestyle='-')
        ax3.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SRO',title_fontsize=18,frameon=False,loc= "upper center")


        ########## labels
        
        ax1.set_xlabel(r't [s]',fontsize=18)
        ax2.set_xlabel(r't [s]',fontsize=18)
        ax3.set_xlabel(r't [s]',fontsize=18)
        ax1.set_ylabel(r'$\nu_e$ luminosity [10$^{51}$ erg.s$^{-1}$]',fontsize=18)
        ax2.set_ylabel(r'$\bar{\nu}_e$ luminosity [10$^{51}$ erg.s$^{-1}$]',fontsize=18)
        ax3.set_ylabel(r'$\nu_x$ luminosity [10$^{51}$ erg.s$^{-1}$]',fontsize=18)


        for i in range(len(ax)-1):
            ax[i].set_xticklabels([])
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.savefig('Nu_lum.png')

    

def Plot_nu_energy(time_range):    #########################  Paper ready Nu energy

        fig,ax = plt.subplots(3,1,figsize=(8,15))
        ax1=ax[0]
        ax2=ax[1]
        ax3=ax[2]


        for base in Shared.all_simulations.keys():
            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['nue_energ'])
            ax1.plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])

            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['nua_energ'])
            ax2.plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])
            
            time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],Shared.all_simulations[base]['nux_energ'])
            ax3.plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])

        ############ limits

        ax1.set_xlim(time_range)
        ax2.set_xlim(time_range)
        ax3.set_xlim(time_range)

        ax1.set_ylim([10.1,20])
        ax2.set_ylim([10.1,20])
        ax3.set_ylim([10.1,20])

        ########### legend

        black_line1, = plt.plot([], [], color='k', linestyle='-')
        black_line2, = plt.plot([], [], color='k', linestyle='--')
        black_line3, = plt.plot([], [], color='k', linestyle=':')
        ax2.legend([black_line1,black_line2,black_line3],[r"Reference",r"$\kappa_a^*$","Full"],fontsize=18,handlelength=1,frameon=False,loc= "upper right")


        black_line1, = plt.plot([], [], color='r', linestyle='-')
        black_line2, = plt.plot([], [], color='orange', linestyle='-')
        ax1.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SFHo',title_fontsize=18,frameon=False,loc= "upper right")
          
        # ~ ax3=ax1.twinx()
        black_line1, = plt.plot([], [], color='b', linestyle='-')
        black_line2, = plt.plot([], [], color='g', linestyle='-')
        ax3.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SRO',title_fontsize=18,frameon=False,loc= "upper center")


        ########## labels
        
        
        ax[-1].set_xlabel(r't [s]',fontsize=18)
        ax1.set_ylabel(r'$\nu_e$ energy [MeV]',fontsize=18)
        ax2.set_ylabel(r'$\bar{\nu}_e$ energy [MeV]',fontsize=18)
        ax3.set_ylabel(r'$\nu_x$ energy [MeV]',fontsize=18)

        for i in range(len(ax)-1):
            ax[i].set_xticklabels([])
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.savefig('Nu_energ.png')





def Plot_Tau(time_range):   #### explo timescales


        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax1=ax

        for base in Shared.all_simulations.keys():
            if "GR1D" in base:
                continue  ###### I got too annoyed trying to find back these values in GR1D files
            else:
                
                tau= (Shared.all_simulations[base]['gain_mass']/Shared.all_simulations[base]['gain_mass_accretion'])/abs(Shared.all_simulations[base]['gain_E_bind']/Shared.all_simulations[base]['gain_net_heating'])
                time,y=Shared.plot_interp(Shared.all_simulations[base]['time'],tau)
                ax1.plot(time-Shared.all_simulations[base]['t_bounce'],y,ls=Shared.all_simulations[base]['ticks'],color=Shared.all_simulations[base]['color'],alpha=Shared.all_simulations[base]['alpha'])
    
    
            ############ limits
    
            ax1.set_xlim(time_range)
    
            ax1.set_ylim([0.,2])
    
            ########### legend
    
            black_line1, = plt.plot([], [], color='k', linestyle='-')
            black_line2, = plt.plot([], [], color='k', linestyle='--')
            black_line3, = plt.plot([], [], color='k', linestyle=':')
            ax1.legend([black_line1,black_line2,black_line3],[r"Reference",r"$\kappa_a^*$","Full"],fontsize=18,handlelength=1,frameon=False,loc= "lower right")
    
            ax2=ax1.twinx()
            black_line1, = plt.plot([], [], color='r', linestyle='-')
            black_line2, = plt.plot([], [], color='orange', linestyle='-')
            ax2.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SFHo',title_fontsize=18,frameon=False,loc= "upper left")
              
            
            ax3=ax1.twinx()
            black_line1, = plt.plot([], [], color='b', linestyle='-')
            black_line2, = plt.plot([], [], color='g', linestyle='-')
            ax3.legend([black_line1,black_line2],[r"OPE",r"T-matrix"],fontsize=18,handlelength=1,title='SRO',title_fontsize=18,frameon=False,loc= "upper center")
    
    
            ########## labels
            
            
            ax1.set_xlabel(r't [s]',fontsize=18)
            ax1.set_ylabel(r'$\tau_{adv}/\tau_{heat}$',fontsize=18)
            
            ax2.set_xticklabels([])    
            ax2.set_yticklabels([])    
            
            ax3.set_xticklabels([])    
            ax3.set_yticklabels([])    
            plt.savefig('Tau.png')
        # ~ plt.show()
    