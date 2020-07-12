import numpy as np
import pandas as pd
from get_coeffs import χ2_of_row, get_masses_and_CKM, minimize_all_rows



v = 246

yu = np.sqrt(2) * 2.16*10**(-3)/v
yc = np.sqrt(2) * 1.27/v
yt = np.sqrt(2) * 173 / v
yd = np.sqrt(2) * 4.7 * 10**(-3)/v
ys = np.sqrt(2) * 96 * 10**(-3) / v
yb = np.sqrt(2) * 4.2/v

ye = np.sqrt(2) * 0.5109989461 * 10**(-3) /v 
yμ = np.sqrt(2) * 105.6583745 * 10**(-3) /v
yτ = np.sqrt(2) * 1776.86 * 10**(-3) /v


σl = 10**(-3)/v * np.sqrt(2) *  np.array([.0000000031, 0.0000024, 0.12] )

yus =  np.array([yu, yc, yt])
yds =  np.array([yd, ys, yb])
yls =  np.array([ye, yμ, yτ])



CKM_exp = np.array( [[.97427, .2243, .00394], [.218, .997, .0422],
           [.0081, .039, 1.019]] )

exp_vals = np.concatenate((yus.flatten(), yds.flatten(), CKM_exp.flatten() , yls.flatten()))

exp_vals_no_CKM = np.concatenate((yus.flatten(), yds.flatten() , yls.flatten()))


σ_CKM =np.array( [[.00021, .0005, .00036], [.004, .017, .0008],
         [.0005, .0023, .025]] )

σ_yu = 10**(-3)/v * np.sqrt(2) * np.array([.5, 20, 400])
σ_yd = 10**(-3)/v * np.sqrt(2) * np.array( [.5, 8, 20] )

cu = np.array([[.3 + .3j, -.2 + .5j, .2 - .2j], [-.3 + .2j, -.4 + .5j, .5 
- .3j], [.4 - .3j, -.4 - .9j, .3 + .5j]])

cd = np.array([[.4 + .2j, .2 - .2j, .2 - .3j], [.1 - .6j, 
    1 - .1j, -.2 - .4j], [-.4 + .3j, -.4 + .6j, .4 + .5j]])

cuR = np.array([.3 ,.3, -.2 , .5, .2 ,- .2, -.3 , .2, -.4 , .5, .5 ,
- .3, .4 ,- .3, -.4 ,-.9, .3 , .5])

cdR = np.array([ 0.4, 0.2,  0.2, -0.2,  0.2, -0.3,  0.1, -0.6,  1. , -0.1, -0.2, -0.4,
       -0.4, 0.3, -0.4, 0.6,  0.4, 0.5])

σs = np.concatenate( (σ_yu.flatten(), σ_yd.flatten(), σ_CKM.flatten(), σl.flatten() ) )
σs_no_CKM = np.concatenate( (σ_yu.flatten(), σ_yd.flatten(),  σl.flatten() ) )

cijNames = ['Re(c11)', 'Im(c11)','Re(c12)', 'Im(c12)','Re(c13)', 'Im(c13)',
            'Re(c21)', 'Im(c21)','Re(c22)', 'Im(c22)','Re(c23)', 'Im(c23)',
            'Re(c31)', 'Im(c31)','Re(c32)', 'Im(c32)','Re(c33)', 'Im(c33)']


cmin = .1
cmax = 1    


#All below is testing for cmax=1 higgs!=0 case

def get_QUDs(n33=1):
    ext = 'unique_Qs_cMax1_n33_'+str(n33)+'.dat'
    df = pd.read_csv('/Users/dillonberger/Dropbox/SU2_U1_flavor_symm/cleaned_code/results/unique_Qs_cMax1/' + ext, 
                      delim_whitespace=True)   
    
    return df


def get_coefs(n33=1, species='up'):
    directory = '''/Users/dillonberger/Dropbox/SU2_U1_flavor_symm/cleaned_code/results/coefficients/n33_'''
    directory = directory + str(n33) + '/'
    file = species + '_coefs.dat'
    coefsDF = pd.read_csv(directory+file, delim_whitespace=True)
    return coefsDF

def get_status(n33=1, species='up'):
    directory = '''/Users/dillonberger/Dropbox/SU2_U1_flavor_symm/cleaned_code/results/coefficients/n33_'''
    directory = directory + str(n33) + '/'
    file = 'ε_&_status.dat'
    statusDF = pd.read_csv(directory+file, delim_whitespace=True)
    return statusDF


def get_all(n33=1):
    upDF = get_coefs(n33, 'up')
    downDF = get_coefs(n33,'down')
    lepDF = get_coefs(n33, 'lep')
    εDF = get_status(n33, 'up')
    QUDs = get_QUDs(n33)
        
    
    return upDF, downDF, lepDF, QUDs, εDF

def structure_coefs_for_line(uCoefs_DF, dCoefs_DF, 
                             lCoefs_DF, ε_DF, line=1):
    uCoefs_for_line = np.array(uCoefs_DF.iloc[line])
    dCoefs_for_line = np.array(dCoefs_DF.iloc[line])
    lCoefs_for_line = np.array(lCoefs_DF.iloc[line])
    ε_for_line = ε_DF.iloc[line]['ε']
    
    structuredCijs = list(np.concatenate((uCoefs_for_line, dCoefs_for_line, lCoefs_for_line)))
    
    return [ε_for_line] + structuredCijs




def prepare_data_for_χ2(upDF, downDF, lepDF, QUDs, εDF, line=1):
    cijList = structure_coefs_for_line(upDF, downDF, lepDF, εDF, line=line)
    QUD = QUDs.iloc[line]
    
    return [cijList, QUD]

def prepare_data_for_masses(upDF, downDF, lepDF, QUDs, εDF, line=1):
    cijs_and_ε = structure_coefs_for_line(upDF, downDF, lepDF, εDF, line=line)
    ε = cijs_and_ε[0]
    cijs = cijs_and_ε[1:]
    QUD = QUDs.iloc[line]
    
    return [QUD, ε, cijs]


def execute(n33, small_χ_only = True):
    upDF, downDF, lepDF, QUDs, εDF = get_all(n33=n33)
    
    χs = []
    χ2_dream =100
    masses = []
    for line in range(len(upDF)):
        if small_χ_only:
            if εDF.iloc[line]['χ2'] > χ2_dream:
                continue 
            
        χ2_dat = prepare_data_for_χ2(upDF, downDF, lepDF, QUDs, εDF, line=line)
        mass_dat = prepare_data_for_masses(upDF, downDF,lepDF, QUDs, εDF, line=line)
        
        χs.append( χ2_of_row(*χ2_dat, transform=False) )
        masses.append( get_masses_and_CKM(*mass_dat) )
        
    return χs, masses
        
        
        
        


for i in range(1,11):
    print('n33 = ' + str(i))
    minimize_all_rows(i, iters=5)                  
        
# χs, masses = execute(1, small_χ_only = True)
    
