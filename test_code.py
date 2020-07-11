#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:41:01 2020

@author: dillonberger
"""



import numpy as np
import pandas as pd
from get_coeffs import χ2_of_row, get_masses_and_CKM, dag, get_ε_matrix



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

upDF = get_coefs(1, 'up')
downDF = get_coefs(1,'down')
lepDF = get_coefs(1, 'lep')
εDF = get_status(1, 'up')
QUDs = get_QUDs(1)

def structure_coefs_for_line(uCoefs_DF, dCoefs_DF, 
                             lCoefs_DF, ε_DF, line=1):
    uCoefs_for_line = np.array(uCoefs_DF.iloc[line])
    dCoefs_for_line = np.array(dCoefs_DF.iloc[line])
    lCoefs_for_line = np.array(lCoefs_DF.iloc[line])
    ε_for_line = ε_DF.iloc[line]['ε']
    
    structuredCijs = list(np.concatenate((uCoefs_for_line, dCoefs_for_line, lCoefs_for_line)))
    
    return [ε_for_line] + structuredCijs



def prepare_data_for_χ2(line):
    cijList = structure_coefs_for_line(upDF, downDF, lepDF, εDF, line=line)
    QUD = QUDs.iloc[line]
    
    return [cijList, QUD]

def prepare_data_for_masses(line):
    cijs_and_ε = structure_coefs_for_line(upDF, downDF, lepDF, εDF, line=line)
    ε = cijs_and_ε[0]
    cijs = cijs_and_ε[1:]
    QUD = QUDs.iloc[line]
    
    return [QUD, ε, cijs]

def test_matrix(QUD, ε, cij_mats, kind='lepton'):

    if kind == 'up':
        cij_mat = cij_mats[0]
    if kind == 'down':
        cij_mat = cij_mats[1]
    if kind == 'lepton':
        cij_mat = cij_mats[2]
        
    εnij = get_ε_matrix(QUD, ε, species = kind)
    Y = cij_mat * εnij
    Ydag = dag(Y)
    YYdag = Y @ Ydag
    m = np.trace(YYdag)/3
    s = np.sqrt( np.trace(YYdag @ YYdag)/3 - m**2 )
    # for i in range(3):
    #     for j in range(3):
    #         if i==j:
    #             continue
    #         else:
    #             tot = tot + YYdag[i,j]
    lbMin = m - s*np.sqrt(2)
    ubMin = m - s/np.sqrt(2)
    lbMax = m + s/np.sqrt(2)
    ubMax = m + s*np.sqrt(2)
    return lbMin, ubMin, lbMax, ubMax
                
        
    

dataLine = prepare_data_for_χ2(8)

z = χ2_of_row(*dataLine)

dataMassesLine = prepare_data_for_masses(8)

# x = sum_off_diags(*dataMassesLine, kind='lepton')

lbMin, ubMin, lbMax, ubMax = test_matrix(*dataMassesLine, kind='up') 

u,d,l,CKM = get_masses_and_CKM(*dataMassesLine)

  
    
    
    





















        


