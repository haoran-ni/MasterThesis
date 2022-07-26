#!/usr/bin/env python
# coding: utf-8

from matplotlib import pylab as plt
from sklearn.metrics import mean_squared_error
from itertools import compress

import xml.etree.ElementTree as ET
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from ase.geometry.analysis import Analysis
from ase.geometry import cellpar_to_cell

from tqdm.notebook import tqdm

import time
import rascal
import json
import math

import ase
from ase.io import read, write
from ase.build import make_supercell
from ase.visualize import view
from ase.geometry import wrap_positions
import numpy as np

from rascal.representations import SphericalInvariants
from rascal.models import Kernel, train_gap_model
from rascal.models.asemd import ASEMLCalculator
from rascal.utils import from_dict, to_dict, CURFilter, dump_obj, load_obj, get_score, print_score, FPSFilter


start = time.time()

def extract_ref(frames,energy_key=True,forces_key=True,number_key=True):
    e,f,n = [], [], []
    for frame in frames:
        if energy_key:
            e.append(frame.info['energy'])
        if forces_key:
            f.append(frame.get_array('forces'))
        if number_key:
            n.append(len(frame))        
    e = np.array(e)
    n = np.array(n)
    try:
        f = np.concatenate(f)
    except:
        pass
    return e,f,n

def EF_baseline(frame, f, r_cut):
    
    '''
    
    f is the function of the baseline, it should take the distance r as the only input. 
    When using CubicSpline() to fit the baseline function f, please set 'extrapolate' = False
    
    r_cut is the cut-off radius, only atoms within this radius will be considered frame is an Atoms object, 
    for each atom in this Atoms object, we calculate its distance to other atoms within the cutoff radius 
    and substract the baseline energy
    
    '''
    
    f_deri = f.derivative() # don't forget the minus sign below
    
    vec_cell = frame.cell.cellpar()
    M1, M2, M3 = vec_cell[0]**2, vec_cell[1]**2, vec_cell[2]**2
    M12 = vec_cell[0]*vec_cell[1]*np.cos(vec_cell[5]*np.pi/180)
    M13 = vec_cell[0]*vec_cell[2]*np.cos(vec_cell[4]*np.pi/180)
    M23 = vec_cell[1]*vec_cell[2]*np.cos(vec_cell[3]*np.pi/180)
    det_M = M1*M2*M3-M1*M23**2-M2*M13**2-M3*M12**2+2*M12*M13*M23
    N1=math.ceil(np.sqrt((M2*M3-M23**2)/det_M)*r_cut)
    N2=math.ceil(np.sqrt((M1*M3-M13**2)/det_M)*r_cut)
    N3=math.ceil(np.sqrt((M1*M2-M12**2)/det_M)*r_cut)
    
    pbc = frame.get_pbc()
    number = len(frame)
    
    cell_size = [(2*N1+1) if pbc[0] else 1,
                  (2*N2+1) if pbc[1] else 1, 
                  (2*N3+1) if pbc[2] else 1]
    center_index = math.floor(np.prod(cell_size)/2)
    il = center_index*number
    ih = (center_index+1)*number
    
    frame2=frame*(cell_size[0],cell_size[1],cell_size[2])

    frc_list = np.zeros((number,3))
    base_energy = 0
    
    dist_idx = np.arange((len(frame2)))
    
    for i in range(il,ih):
        i_energy=0
        i_distances = frame2.get_distances(i,dist_idx)
        for j in dist_idx:
            if str(f(i_distances[j])) != 'nan':
                if center_index*number <= j < (center_index+1)*number:
                    i_energy += float(f(i_distances[j]))/2
                    frc_norm = -f_deri(i_distances[j])
                    rel_vec = normalize( frame2[i].position - frame2[j].position )
                    frc_vec = frc_norm * rel_vec
                    frc_list[i-il,0] += frc_vec[0]
                    frc_list[i-il,1] += frc_vec[1]
                    frc_list[i-il,2] += frc_vec[2]
                else:
                    i_energy += float(f(i_distances[j]))
                    frc_norm = -f_deri(i_distances[j])
                    rel_vec = normalize( frame2[i].position - frame2[j].position )
                    frc_vec = frc_norm * rel_vec
                    frc_list[i-il,0] += frc_vec[0]
                    frc_list[i-il,1] += frc_vec[1]
                    frc_list[i-il,2] += frc_vec[2]
            else:
                pass    
        base_energy += i_energy
    return base_energy, frc_list

def normalize(v):
    '''
    v is a numpy array
    '''
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    else:
        return v / norm


### R6 model

tree = ET.parse('P_r6_innercut.xml')
root = tree.getroot()


r=[]
energy=[]
for i in root.iter('potential_pair'):
    for j in i.iter():
        if j.tag != 'potential_pair':
            r.append(float(j.attrib['r']))
            energy.append(float(j.attrib['E']))
        else:
            pass

energy=np.array(energy)/2
r=np.array(r)

r_new = np.linspace(2.8,20,500)

### base_func is the ultimate baseline function! Its domain is [3,20].

base_func = CubicSpline(r, energy, bc_type = 'clamped',extrapolate = False)


reg_e_dic={'P2/P4':0.03,
         'rss_200':0.035,
         'rss_005':0.035,
         '2D':0.01,
         'rss_3c':0.025,
         'cryst_dist':0.03,
         'liq_12_03_02_network':0.03,
         'rss_rnd':0.05,
         'liq_12_03_01_liqP4':0.4,
         'phosphorene':0.03,
         'phosphorus_ribbons':0.03,
         'isolated_atom':0.04}

reg_f_dic={'P2/P4':0.4,
         'rss_200':0.4,
         'rss_005':0.4,
         '2D':0.07,
         'rss_3c':0.4,
         'cryst_dist':0.3,
         'liq_12_03_02_network':0.4,
         'rss_rnd':0.35,
         'liq_12_03_01_liqP4':0.5,
         'phosphorene':0.4,
         'phosphorus_ribbons':0.4,
         'isolated_atom':0.6}

### Training parameters
param2bd={
    'cutoff':5.,
    'nmax':12,
    'lmax':1,
    'norm':False,
    'nrep':15,
    'zeta':1,
    'ereg':0.003,
    'freg':0.003
}

paramSOAP={
    'cutoff':5.,
    'nmax':12,
    'lmax':6,
    'norm':True,
    'nrep':8000,
    'zeta':4,
    'ereg':0.03,
    'freg':0.4
}

print('2bd training parameters:')
print(param2bd)
print('')
print('SOAP training parameters:')
print(paramSOAP)
print('')
print('Regularizors:')
print(reg_e_dic)
print(reg_f_dic)
print('')

#### 2bd + SOAP + R6

N = 4798
N_exf = 100
frames_train = read('training_data_no_baseline.xyz', index=':{}'.format(N))
frames_exf = read('exfoliation_mbd_no_baseline.xyz', index=':{}'.format(N_exf))

np.random.seed(10)
np.random.shuffle(frames_train)

self_contributions = {
    15: -0.0975258129554152  #-5.157   
}

y_train, f_train, n_train = extract_ref(frames_train,True,True,True)


### 2bd training

hypers_2bd = dict(soap_type="RadialSpectrum",
              interaction_cutoff = param2bd['cutoff'], 
              max_radial = param2bd['nmax'], 
              max_angular = param2bd['lmax'], 
              gaussian_sigma_constant = 0.5,
              gaussian_sigma_type = "Constant",
              cutoff_smooth_width = 1.0,
              normalize = param2bd['norm'],
              radial_basis = "GTO",
              compute_gradients = True,
              expansion_by_species_method = 'structure wise')

soap_2bd = SphericalInvariants(**hypers_2bd)

managers_train_2bd = soap_2bd.transform(frames_train)
managers_exf_2bd = soap_2bd.transform(frames_exf)

n_sparse = {15:param2bd['nrep']}

compressor = FPSFilter(soap_2bd, n_sparse, act_on='sample per species')
X_sparse = compressor.select_and_filter(managers_train_2bd)

zeta = param2bd['zeta']
kernel = Kernel(soap_2bd, name='GAP', zeta=zeta, target_type='Structure', kernel_type='Sparse')

KNM = kernel(managers_train_2bd, X_sparse)
KNM_down = kernel(managers_train_2bd, X_sparse, grad=(True, False))
KNM = np.vstack([KNM, KNM_down])
del KNM_down

model_2bd = train_gap_model(kernel, frames_train, KNM, X_sparse, y_train, self_contributions, 
                        grad_train=-f_train, lambdas=[param2bd['ereg'],param2bd['freg']], jitter=1e-13)


y_pred_train = model_2bd.predict(managers_train_2bd)
f_pred_train = model_2bd.predict_forces(managers_train_2bd)
y_2bd_pred_exf = model_2bd.predict(managers_exf_2bd)


y_train = y_train - y_pred_train

f_train = f_train - f_pred_train


self_contributions = {
    15: 0.0
}


### SOAP training

hypers = dict(soap_type="PowerSpectrum",
              interaction_cutoff = paramSOAP['cutoff'], 
              max_radial = paramSOAP['nmax'], 
              max_angular = paramSOAP['lmax'], 
              gaussian_sigma_constant = 0.5,
              gaussian_sigma_type = "Constant",
              cutoff_smooth_width = 1.0,
              normalize = paramSOAP['norm'],
              radial_basis = "GTO",
              compute_gradients = True,
              expansion_by_species_method = 'structure wise',
              )

soap = SphericalInvariants(**hypers)

managers_train = soap.transform(frames_train)
managers_exf_soap = soap.transform(frames_exf)

n_sparse = {15:paramSOAP['nrep']}

compressor = FPSFilter(soap, n_sparse, act_on='sample per species')
X_sparse = compressor.select_and_filter(managers_train)

zeta = paramSOAP['zeta']
kernel = Kernel(soap, name='GAP', zeta=zeta, target_type='Structure', kernel_type='Sparse')

KNM = kernel(managers_train, X_sparse)
KNM_down = kernel(managers_train, X_sparse, grad=(True, False))
KNM = np.vstack([KNM, KNM_down])
del KNM_down


num = 0
for frame in frames_train:
    num += len(frame)

reg_e = np.zeros((N,1))
reg_f = []

for i in np.arange(N):
    try:
        name=frames_train[i].info['config_type']
        reg_e[i]=reg_e_dic[name]

        n_lines = 3*len(frames_train[i])
        val = reg_f_dic[name]
        for ii in range(n_lines):
            reg_f.append(val)
    except:
        reg_e[i]=reg_e_dic['P2/P4']

        n_lines = 3*len(frames_train[i])
        val = reg_f_dic['P2/P4']
        for ii in range(n_lines):
            reg_f.append(val)

reg_f = np.array(reg_f).reshape((3*num,1))


modelSOAP = train_gap_model(kernel, frames_train, KNM, X_sparse, y_train, self_contributions, 
                        grad_train=-f_train, lambdas=[reg_e,reg_f], jitter=1e-13)

y_soap_pred_exf = modelSOAP.predict(managers_exf_soap)

y_pred = y_2bd_pred_exf + y_soap_pred_exf

for i in np.arange(len(y_pred)):
    y_pred[i] += EF_baseline(frames_exf[i], base_func, 20.)[0]

print('2bd+SOAP+R6 finished!')
del KNM
del modelSOAP
del model_2bd
del frames_train
del frames_exf
del managers_train
del managers_train_2bd
del managers_exf_2bd
del managers_exf_soap
del compressor
del X_sparse
del kernel

#### 2bd + SOAP

frames_train = read('P_GAP_20_fitting_data.xyz', index=':{}'.format(N))
frames_exf = read('exfoliation_mbd_reference.xyz', index=':{}'.format(N_exf))

for frame in frames_train:
    frame.set_positions(wrap_positions(frame.get_positions(),frame.get_cell(),eps=1e-10))
for frame in frames_exf:
    frame.set_positions(wrap_positions(frame.get_positions(),frame.get_cell(),eps=1e-10))

np.random.seed(10)
np.random.shuffle(frames_train)

y_train, f_train, n_train = extract_ref(frames_train,True,True,True)

self_contributions = {
    15: -0.0975258129554152  #-5.157   
}

### 2bd training

soap_2bd = SphericalInvariants(**hypers_2bd)

managers_train_2bd = soap_2bd.transform(frames_train)
managers_exf_2bd = soap_2bd.transform(frames_exf)

n_sparse = {15:param2bd['nrep']}

compressor = FPSFilter(soap_2bd, n_sparse, act_on='sample per species')
X_sparse = compressor.select_and_filter(managers_train_2bd)

zeta = param2bd['zeta']
kernel = Kernel(soap_2bd, name='GAP', zeta=zeta, target_type='Structure', kernel_type='Sparse')

KNM = kernel(managers_train_2bd, X_sparse)
KNM_down = kernel(managers_train_2bd, X_sparse, grad=(True, False))
KNM = np.vstack([KNM, KNM_down])
del KNM_down

model_2bd = train_gap_model(kernel, frames_train, KNM, X_sparse, y_train, self_contributions, 
                        grad_train=-f_train, lambdas=[param2bd['ereg'],param2bd['freg']], jitter=1e-13)

y_pred_train = model_2bd.predict(managers_train_2bd)
f_pred_train = model_2bd.predict_forces(managers_train_2bd)
y_2bd_pred_exf = model_2bd.predict(managers_exf_2bd)


y_train = y_train - y_pred_train

f_train = f_train - f_pred_train


self_contributions = {
    15: 0.0
}


### SOAP training

soap = SphericalInvariants(**hypers)

managers_train = soap.transform(frames_train)
managers_exf_soap = soap.transform(frames_exf)

n_sparse = {15:paramSOAP['nrep']}

compressor = FPSFilter(soap, n_sparse, act_on='sample per species')
X_sparse = compressor.select_and_filter(managers_train)

zeta = paramSOAP['zeta']
kernel = Kernel(soap, name='GAP', zeta=zeta, target_type='Structure', kernel_type='Sparse')

KNM = kernel(managers_train, X_sparse)
KNM_down = kernel(managers_train, X_sparse, grad=(True, False))
KNM = np.vstack([KNM, KNM_down])
del KNM_down

modelSOAP = train_gap_model(kernel, frames_train, KNM, X_sparse, y_train, self_contributions, 
                        grad_train=-f_train, lambdas=[reg_e,reg_f], jitter=1e-13)

y_soap_pred_exf = modelSOAP.predict(managers_exf_soap)

y_pred_2 = y_2bd_pred_exf + y_soap_pred_exf

print('2bd+SOAP finished!')



#### Plots

dist_list = []
e_list = []

for frame in frames_exf:
    dist = frame.get_positions()[0,1]-frame.get_positions()[6,1]
    e = frame.info['energy']
    dist_list.append(dist)
    e_list.append(e)
    
fig,ax = plt.subplots(figsize=(7,5))

ax.set_xlabel('Interlayer distance (Angstrom)',fontsize=20)
ax.set_ylabel('Energy (eV)',fontsize=20)
ax.tick_params(labelsize=16)

ax.plot(dist_list, e_list,'-',label='DFT + MBD energy',linewidth=5,color='blue')
ax.plot(dist_list, y_pred_2,'--',label='2bd + SOAP prediction',linewidth=4,color='g')
ax.plot(dist_list, y_pred,'--',label='2bd + SOAP + R6 prediction',linewidth=4,color='r')
ax.legend(markerscale=2,fontsize=18)

plt.tight_layout()


fig.savefig('./figs/exfoliation.png')


end = time.time()

print('')
print("The time of execution is :", end-start)

