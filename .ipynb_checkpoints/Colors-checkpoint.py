
import Shared
import matplotlib.pylab as plt
import numpy as np

def Set_colors_and_ticks():
    for base in Shared.all_simulations.keys():
        
        if  (("per" in base) or ("hr" in base) or ("nr" in base) or ("nb" in base) or ("new" in base)):
            Shared.all_simulations[base]['alpha']=0
            
        elif ('SFHo' in base ) & ('8.dat' in base) :
            Shared.all_simulations[base]['alpha']=1.
        elif  ('SFHo' in base ) & ( '7.dat' in base) :
            Shared.all_simulations[base]['alpha']=0
        elif ('SRO' in base) & ('8' in base) :
            Shared.all_simulations[base]['alpha']=0.
        elif  ('SRO' in base)  :
            Shared.all_simulations[base]['alpha']=1.
        else: 
            Shared.all_simulations[base]['alpha']=1.
            
        if 'ref' in base: 
            Shared.all_simulations[base]['ticks']='-'
        elif 'e8' in base: 
            Shared.all_simulations[base]['ticks']='-'
        else:
            Shared.all_simulations[base]['ticks']='--'
            
        if 'ref' in base: 
            Shared.all_simulations[base]['marker']='o'
        elif 'e8' in base or 'ax_on' in base: 
            Shared.all_simulations[base]['marker']='D'
        else:
            Shared.all_simulations[base]['marker']='^'
            
        if ('e8' in base) or ('ax_on' in base):
            col = plt.cm.jet(np.linspace(1,0,7))
            if 'ref' in base: 
                Shared.all_simulations[base]['color']=col[5]
            elif '1_5e8' in base:
                Shared.all_simulations[base]['color']=col[1]
            elif '1e8' in base:
                Shared.all_simulations[base]['color']=col[0]
            elif '2e8' in base:
                Shared.all_simulations[base]['color']=col[2]
            elif '3e8' in base:
                Shared.all_simulations[base]['color']=col[3]
            elif '5e8' in base:
                Shared.all_simulations[base]['color']=col[4]
            else:
            Shared.all_simulations[base]['color']='k'
            print("case not found")

                
        elif ('Gang' in base) & ('SFHo' in base) : 
            Shared.all_simulations[base]['color']='orange'
        elif 'nr' in base :
            Shared.all_simulations[base]['color']='m'
        elif 'SFHo' in base :
            Shared.all_simulations[base]['color']='r'
        elif ('Gang' in base) & ('SRO' in base) : 
            Shared.all_simulations[base]['color']='g'
        elif 'SRO' in base :
            Shared.all_simulations[base]['color']='b'
          
        else:
            Shared.all_simulations[base]['color']='k'
            print("case not found")
            