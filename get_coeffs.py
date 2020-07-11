#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:45:53 2020

@author: dillonberger
"""

import numpy as np 
import pandas as pd
import os


from det_checker import test_charges, lower_bound_filter, namesOut,namesIn,flatten

from main import structure_det_filtered



import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.lines import Line2D
from scipy.optimize import minimize, Bounds
from numpy.linalg import eigh

import random as rnd






from matplotlib import rc
rc('text', usetex=True)


cmap = cm.get_cmap('rainbow', 3)


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




yus =  np.array([yu, yc, yt])
yds =  np.array([yd, ys, yb])
yls =  np.array([ye, yμ, yτ])



CKM_exp = np.array( [[.97427, .2243, .00394], [.218, .997, .0422],
           [.0081, .039, 1.019]] )

exp_vals = np.concatenate((yus.flatten(), yds.flatten(), CKM_exp.flatten() , yls.flatten()))

exp_vals_no_CKM = np.concatenate((yus.flatten(), yds.flatten() , yls.flatten()))


σ_CKM =np.array( [[.00021, .0005, .00036], [.004, .017, .0008],
         [.0005, .0023, .025]] ).flatten()

σ_yu = 10**(-3)/v * np.sqrt(2) * np.array([.5, 20, 400])
σ_yd = 10**(-3)/v * np.sqrt(2) * np.array( [.5, 8, 20] )
σ_yl = 10**(-3)/v * np.sqrt(2) *  np.array([.0000000031, 0.0000024, 0.12] )

cu = np.array([[.3 + .3j, -.2 + .5j, .2 - .2j], [-.3 + .2j, -.4 + .5j, .5 
- .3j], [.4 - .3j, -.4 - .9j, .3 + .5j]])

cd = np.array([[.4 + .2j, .2 - .2j, .2 - .3j], [.1 - .6j, 
    1 - .1j, -.2 - .4j], [-.4 + .3j, -.4 + .6j, .4 + .5j]])

cuR = np.array([.3 ,.3, -.2 , .5, .2 ,- .2, -.3 , .2, -.4 , .5, .5 ,
- .3, .4 ,- .3, -.4 ,-.9, .3 , .5])

cdR = np.array([ 0.4, 0.2,  0.2, -0.2,  0.2, -0.3,  0.1, -0.6,  1. , -0.1, -0.2, -0.4,
       -0.4, 0.3, -0.4, 0.6,  0.4, 0.5])

# σs = np.concatenate( (σ_yu.flatten(), σ_yd.flatten(), σ_CKM.flatten(), σl.flatten() ) )
# σs_no_CKM = np.concatenate( (σ_yu.flatten(), σ_yd.flatten(),  σl.flatten() ) )

cijNames = ['Re(c11)', 'Im(c11)','Re(c12)', 'Im(c12)','Re(c13)', 'Im(c13)',
            'Re(c21)', 'Im(c21)','Re(c22)', 'Im(c22)','Re(c23)', 'Im(c23)',
            'Re(c31)', 'Im(c31)','Re(c32)', 'Im(c32)','Re(c33)', 'Im(c33)']


cmin = .1
cmax = 1            

np.seterr(invalid='raise')
            


def nu(i,j, hu=1, Q=[], uR=[]):
    return( np.abs(Q[i]- uR[j] - hu ) )

def nd(i,j, hd=1, Q=[], dR=[]):
    return( np.abs(Q[i]- dR[j] - hd ) )

def ne(i,j, hd=1, L=[], eR=[]):
    return( np.abs(L[i]- eR[j] - hd ) )

def dag(mat):
    return np.conj(np.transpose(mat))


def chop_mat(mat, max=10**(-15)):
    dims = np.shape(mat)
    new_mat = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            if np.abs(mat[i,j]) < max:
                new_mat[i,j] = 0
            else:
                new_mat[i,j] = mat[i,j]
                
    return new_mat

def chop_list(lst, max=10**(-15)):
    new_mat = np.zeros(len(lst))
    for i in range(len(lst)):
        if np.abs(lst[i]) < max:
            new_mat[i] = 0
        else:
            new_mat[i] = lst[i]
            
    return new_mat
    
    

def read(n33=1):
    ext = 'unique_Qs_cMax1_n33_'+str(n33)+'.dat'
    df = pd.read_csv('/Users/dillonberger/Dropbox/SU2_U1_flavor_symm/cleaned_code/results/unique_Qs_cMax1/' + ext, 
                      delim_whitespace=True)   
    
    return df



def get_bounds(n33=1, cijMax=1, cijMin=.1, higgs_zero=False):
    
    if higgs_zero:
        dirname = 'higgs0_' + 'cijMax'
    else:
        dirname = 'cijMax'    
        
   
    dirname = 'S_complex_' + dirname
    folder = '/Users/dillonberger/Dropbox/SU2_U1_flavor_symm/cleaned_code/results/'+dirname+str(cijMax)
    
    det_filtered = pd.read_csv(folder +'/det_filtered2/n33_'+str(n33)+'_charges.dat', 
                      header=0, names=namesOut, delim_whitespace=True)  
    det_QUDs = structure_det_filtered(det_filtered)     
    
    good_QUDs, l_bounds, which_bounds = lower_bound_filter(det_QUDs, cijMax,
                                               n33=n33, 
                                               bounds_fname= folder+'/poly_filtered2/bounds_n33_'+str(n33)+'.dat')

    min_lbound = min(l_bounds)
    u_bound = (yτ/cmin)**(1/n33)
    
    return min_lbound, u_bound
    


    
def get_ε_matrix(QUD, ε, species='up'):
    Q = [QUD['Q1'], QUD['Q2'], QUD['Q3']]
    uR = [QUD['U1'], QUD['U2'], QUD['U3']]
    dR = [QUD['d1'], QUD['d2'],QUD['d3']]
    L = [QUD['L1'], QUD['L2'],QUD['L3']]
    eR = [QUD['eR1'], QUD['eR2'],QUD['eR3']]
    hu = QUD['hu']
    hd = QUD['hd']
    
    
    if species == 'up':
        n = nu
        kwargs = {'hu': hu , 'uR' :uR, 'Q': Q}
    if species == 'down':
        n = nd
        kwargs = {'hd': hd , 'dR': dR, 'Q': Q}
    if species == 'lepton':
        n = ne
        kwargs = {'hd': hd , 'eR': eR, 'L': L}
    
    matrix = np.zeros((3,3))
    
    for i in range(3):
        for j in range(3):
            matrix[i,j] = ε**( n(i,j,**kwargs) )
            
    return matrix
    



def list_to_mat(cij_list):    
    complex_cijs = []
    for i in range(0,len(cij_list),2):
        new_ele = cij_list[i] + 1j*cij_list[i+1]
        complex_cijs.append(new_ele)           
    cij_mat = np.reshape(complex_cijs, (3,3))
    
    return cij_mat

def mat_to_list(cij_mat):
    cij_mat = np.array(cij_mat).flatten()
    cij_list = np.zeros(2*len(cij_mat))
    k = 0
    for i in range(len(cij_mat)):
            cij_list[k] = cij_mat[i].real
            cij_list[k+1] = cij_mat[i].imag
            k+=2
            
    return cij_list
        
    
    


def get_masses_and_CKM(QUD, ε, cij_mats):

    kinds = ['up', 'down', 'lepton']
    eVal_list = []
    for kind in kinds:
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
        
        

        eVals, eVecs = eigh(YYdag) 

        # eVals = chop_list(eVals)
        
        
        
        eVal_list.append( np.sqrt(np.abs(eVals)) )
  
        
        if kind == 'down':
            UL_down = eVecs
        # already checked to make sure returns correct mass 
            
        if kind == 'up':
            UL_up_dag = dag(eVecs)
        # already checked to make sure returns correct mass             

        
    V_CKM = UL_up_dag @ UL_down
    
    upYuks = eVal_list[0]
    downYuks = eVal_list[1]
    lepYuks = eVal_list[2]
    
    return upYuks, downYuks, lepYuks, np.abs(V_CKM)
        
    



def χ2_of_row(allCijs, QUD=[]):
    
    ε = allCijs[0]
    cijs = allCijs[1:]
    cij_ulist = cijs[0:18]
    cij_dlist = cijs[18: 2*18]
    cij_llist = cijs[2*18: 3*18]
    cij_lists = [cij_ulist, cij_dlist, cij_llist]
    
    cij_mats = []
    for cij_list in cij_lists:
        cij_mat = list_to_mat(cij_list)
        cij_mats.append(cij_mat)
    
    upYuks, downYuks, lepYuks, V_CKM = get_masses_and_CKM(QUD, ε, cij_mats)
    
    CKM = V_CKM.flatten()
    V_CKM_exp = CKM_exp.flatten()
    
    χ2_up = (1/3) * sum( np.sqrt((upYuks - yus)**2)/σ_yu)
    χ2_down = (1/3) * sum( np.sqrt((downYuks - yds)**2)/σ_yd )
    χ2_leps = (1/3) * sum( np.sqrt((lepYuks - yls)**2) /σ_yl)
    χ2_CKM = (1/9) * sum( np.sqrt((CKM - V_CKM_exp)**2)/σ_CKM )
    
    χ2 = (1/4) * (χ2_up + χ2_down + χ2_leps + χ2_CKM)
    
    # χ2 less than one means each is less than one
    
    return χ2
    



def find_min_of_row(ε_cijs, QUD, εmin, εmax):

    
    #upper and lower bound is n33 dep
    cons = [{'type': 'ineq', 'fun': lambda x:  x[0] - εmin },
            {'type': 'ineq', 'fun': lambda x:  εmax - x[0] },
            {'type': 'ineq', 'fun': lambda x:  x[0] }]
    

    
    for i in range(1,len(ε_cijs),2):
        
        cons.append({'type': 'ineq', 'fun': lambda x:  np.sqrt(x[i]**2 + x[i+1]**2) - cmin})
        cons.append({'type': 'ineq', 'fun': lambda x: cmax - np.sqrt(x[i]**2 + x[i+1]**2) })


    
    
    
    min_χ = minimize(χ2_of_row,ε_cijs, args=(QUD), 
                     method='trust-constr', constraints = cons, 
                     options={'maxiter': 5000}
                      
                     )
    # print(min_χ)
    res = min_χ.x
    ε = res[0]
    cijs = res[1:]
    
    cij_ulist = cijs[0:18]
    cij_dlist = cijs[18: 2*18]
    cij_llist = cijs[2*18: 3*18]
    
    uCijs = list_to_mat(cij_ulist)
    dCijs = list_to_mat(cij_dlist)
    lCijs = list_to_mat(cij_llist)
    

    
    return uCijs, dCijs, lCijs, ε





def generate_random_coefs(length=18):
    lst = []
    for i in range(length):
        if i%2 == 0:
            randmag = rnd.uniform(cmin, cmax)
        else: 
            randmag = rnd.uniform(.001, 
                                  np.sqrt(cmax**2 - randmag**2))


            
        randsign = rnd.choice([-1,1])
        lst.append(randmag*randsign)
        
    return lst


def test_mat_ord_1(mat):
    violation_count = 0
    for i in range(3):
        for j in range(3):
            if np.abs(mat[i,j]) > cmax:
                violation_count += 1
            if np.abs(mat[i,j]) < cmin:
                violation_count+=1
            
    return violation_count

def run_til_small_for_row(QUD, εmin, εmax, iters=10):
    
    
    #need to get these boudns explicitly
    ε0 = (εmin + εmax)/2
    allCijs = [ε0] + list(np.concatenate((cuR, cdR, cdR)))
    i = 0
    
    trigger = True
    violations_min = 100
    while trigger:
        print(i)
        if i == 0:
            ε_cijs = allCijs
        else:
            c0u = generate_random_coefs()
            c0d = generate_random_coefs()
            c0l = generate_random_coefs()
            ε_cijs = [ε0] + list(np.concatenate((c0u, c0d, c0l)))
            
            
             
            
        uCijs, dCijs, lCijs, ε = find_min_of_row(ε_cijs,QUD, εmin, εmax)
        
        CijMats = [uCijs, dCijs, lCijs]
        violations_count = 0
        for CijMat in CijMats:
            violations_count += test_mat_ord_1(CijMat)
        
        if violations_count != 0:
            if violations_count < violations_min:
                violations_min = violations_count
                minU, minD, minL, εmin = uCijs, dCijs, lCijs, ε
            if i == iters:
                print('No solution found after ' + str(i) + ''' iterations. 
                      Best found has ''' + str(violations_min) + ' violations')
                return minU, minD, minL, εmin, violations_min
            i+=1
            trigger = True
            continue   
        else:
            print('Found Solution: ' + str(violations_count) + ' violations ')
            return uCijs, dCijs, lCijs, ε, violations_count
                
 


   
    
          
def minimize_all_rows(n33=1, only_ord_one=True, iters = 5):
    QUDs = read(n33)
    εmin, εmax = get_bounds(n33)
    
    uDF = pd.DataFrame(columns = cijNames)
    dDF = pd.DataFrame(columns = cijNames)
    lDF = pd.DataFrame(columns = cijNames)
    εDF = pd.DataFrame(columns = ['ε', 'O(1)_violations'])
    
    numFound = 0
    for i in range(len(QUDs)):
        print( 'On line: ' + str(i) )
        QUD = QUDs.iloc[i]
        uCijs, dCijs, lCijs, ε, num_violations = run_til_small_for_row(QUD, εmin, εmax, iters=iters)
          
        
        uDF.loc[i] = mat_to_list(uCijs)
        dDF.loc[i] = mat_to_list(dCijs)
        lDF.loc[i] = mat_to_list(lCijs)
        εDF.loc[i] = np.array([ε, int(num_violations)])
        

        
        udl = ['up_coefs.dat', 'down_coefs.dat', 'lep_coefs.dat']
        udlDFs = [uDF, dDF, lDF]
        path = '/Users/dillonberger/Dropbox/SU2_U1_flavor_symm/cleaned_code/results/coefficients/'
        folder = 'n33_' + str(n33) + '/'
        
        for i in range(len(udl)):
            udlDFs[i].to_csv(path + folder + udl[i],index=False,
                             sep=' ')
        
        εDF.to_csv(path + folder + 'ε_&_status.dat',
                   index= False, sep= ' ')
        
        if num_violations == 0:
            numFound +=1
        
        if numFound == 2:
            return numFound
        

    
            
    return numFound
       

for i in range(1,11):
    minimize_all_rows(i, iters=5)     
        
        
    
    

# QUD = read(1).iloc[0]
# print(QUD)
# εmin, εmax = get_bounds(1)
# u,d,l, ε = run_til_small_for_row(QUD, εmin, εmax, iters=1)


        
    
    
    

    
    
    
    
    
    
    
    