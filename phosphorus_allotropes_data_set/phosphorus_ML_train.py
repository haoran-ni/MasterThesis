#!/usr/bin/env python
# coding: utf-8


# This file is the training of the ML model on all 4798 phosphorus frames, it should be ran on HPC
# This file will output the parity plots of the energies and forces of the training set and the test set
# The model used here is the 2bd+SOAP+R6 model
# We can easily switch to the 2bd+SOAP model by using 'P_GAP_20_fitting_data.xyz' as the training set and 'P_test_set.xyz' as the test set
# Here we are using different regularizers for different structures, we can change the regularizers by changing 'reg_e_dic' and 'reg_f_dic'.


from matplotlib import pylab as plt
from sklearn.metrics import mean_squared_error
from itertools import compress

from ase.geometry.analysis import Analysis
from ase.geometry import cellpar_to_cell
import math

from tqdm.notebook import tqdm

import time
import rascal
import json
import math

import ase
from ase.io import read, write
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
    'ereg':reg_e_dic,
    'freg':reg_f_dic
}

print('2bd training parameters:')
print(param2bd)
print('')
print('SOAP training parameters:')
print(paramSOAP)
print('')

N = 4798
N2 = 1601
frames_train = read('training_data_no_baseline.xyz', index=':{}'.format(N))
frames_test = read('test_data_no_baseline.xyz', index=':{}'.format(N2))


np.random.seed(10)
np.random.shuffle(frames_train)

print('Sorting training set...')
dataset={'P2/P4':[]}
for i in range(N):
    try:
        name=frames_train[i].info['config_type']
        if name in dataset:
            dataset[name].append(i)
        else:
            dataset[name]=[i]
    except:
        dataset['P2/P4'].append(i)

print('Sorting test set...')
dataset2={'P2/P4':[]}
for i in range(N2):
    try:
        name=frames_test[i].info['config_type']
        if name in dataset2:
            dataset2[name].append(i)
        else:
            dataset2[name]=[i]
    except:
        dataset2['P2/P4'].append(i)

del dataset['isolated_atom']
dataset2['liq_12_03_01_liqP4'] = dataset2['liq_12_03_01_liqP4'] + dataset2['liq_P4']
del dataset2['liq_P4']
del dataset2['P2/P4']  # There are no P2/P4 frames in the test set


self_contributions = {
    15: -0.0975258129554152  #-5.157   
}

y_train, f_train, n_train = extract_ref(frames_train,True,True,True)
y_test, f_test, n_test = extract_ref(frames_test,True,True,True)


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
managers_test_2bd = soap_2bd.transform(frames_test)

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



### Correlation plots of the training set using 2bd model

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,12))

ax[0,0].set_xlim(-23,23)
ax[0,0].set_ylim(-23,23)
ax[0,1].set_xlim(-12,12)
ax[0,1].set_ylim(-12,12)
ax[1,0].set_xlim(-6,1)
ax[1,0].set_ylim(-6,1)
ax[1,1].set_xlim(-5.5,-4)
ax[1,1].set_ylim(-5.5,-4)
ax[0,0].plot([0, 1], [0, 1], transform=ax[0,0].transAxes,color='k',linewidth=1)
ax[0,1].plot([0, 1], [0, 1], transform=ax[0,1].transAxes,color='k',linewidth=1)
ax[1,0].plot([0, 1], [0, 1], transform=ax[1,0].transAxes,color='k',linewidth=1)
ax[1,1].plot([0, 1], [0, 1], transform=ax[1,1].transAxes,color='k',linewidth=1)

fig.suptitle('correlation plots of training set',fontsize='x-large')
ax[0,0].set_title('GAP-RSS generated data',fontsize='x-large')
ax[0,0].set_xlabel('reference forces',fontsize='x-large')
ax[0,0].set_ylabel('predicted forces',fontsize='x-large')
ax[0,1].set_title('manually constructed data',fontsize='x-large')
ax[0,1].set_xlabel('reference forces',fontsize='x-large')
ax[0,1].set_ylabel('predicted forces',fontsize='x-large')

ax[1,0].set_title('GAP-RSS generated data',fontsize='x-large')
ax[1,0].set_xlabel('reference energy',fontsize='x-large')
ax[1,0].set_ylabel('predicted energy',fontsize='x-large')
ax[1,1].set_title('manually constructed data',fontsize='x-large')
ax[1,1].set_xlabel('reference energy',fontsize='x-large')
ax[1,1].set_ylabel('predicted energy',fontsize='x-large')

RSS_list = ['rss_rnd', 'rss_005', 'rss_200', 'rss_3c']
manual_list = ['liq_12_03_01_liqP4', 'liq_12_03_02_network', 'phosphorene', 'phosphorus_ribbons','2D','cryst_dist','cryst_dist_hp']

for key in dataset.keys():
    frames_temp = [frames_train[i] for i in dataset[key]]
    y_temp, f_temp, n_temp = extract_ref(frames_temp,True,True,True)
    y_temp_per_atom = y_temp/n_temp
    managers_temp_2bd = managers_train_2bd.get_subset(dataset[key])
    y_pred_temp = model_2bd.predict(managers_temp_2bd) 
    y_pred_temp_per_atom = y_pred_temp/n_temp
    f_pred_temp = model_2bd.predict_forces(managers_temp_2bd)
    
    x = f_temp.flatten()
    y = f_pred_temp.flatten()
    
    if key in RSS_list:
        if key == 'rss_3c':
            order = 10
            color = 'green'
            label = 'RSS, 3-coordinated'
        elif key == 'rss_200':
            order = 8
            color = 'lime'
            label = 'RSS, relaxed'
        elif key == 'rss_005':
            order = 6
            color = 'aqua'
            label = 'RSS, intermediates'
        elif key == 'rss_rnd':
            order = 4
            color = 'dodgerblue'
            label = 'RSS, initial (random)'
        ax[0,0].scatter(x, y, s=6,c=color,marker='o',label=label,zorder=order)
        ax[1,0].scatter(y_temp_per_atom, y_pred_temp_per_atom, s=6,c=color,marker='o',label=label,zorder=order)
        
        rmse_e_temp = mean_squared_error(y_pred_temp_per_atom, y_temp_per_atom, squared=False)
        rmse_f_temp = mean_squared_error(y, x, squared=False) 
        print(str(key)+':')
        print('Forces RMSE: ' + str(rmse_f_temp))
        print('Energy RMSE: ' + str(rmse_e_temp))
        print('')
        
    elif key in manual_list:
        if key == 'cryst_dist':
            order = 10
            color = 'green'
            label = 'Bulk crystals'
        elif key == 'phosphorene' or key == 'phosphorus_ribbons':
            order = 8
            color = 'lime'
            label = None
        elif key == '2D':
            order = 8
            color = 'lime'
            label = '2D structures'
        elif key == 'liq_12_03_01_liqP4':
            order = 6
            color = 'aqua'
            label = 'Molecular liquid'
        elif key == 'liq_12_03_02_network':
            order = 4
            color = 'dodgerblue'
            label = 'Network liquid'
        ax[0,1].scatter(x, y, s=6,c=color,marker='o',label=label,zorder=order)
        ax[1,1].scatter(y_temp_per_atom, y_pred_temp_per_atom, s=6,c=color,marker='o',label=label,zorder=order)
        
        rmse_e_temp = mean_squared_error(y_pred_temp_per_atom, y_temp_per_atom, squared=False)
        rmse_f_temp = mean_squared_error(y, x, squared=False) 
        print(str(key)+':')
        print('Forces RMSE: ' + str(rmse_f_temp))
        print('Energy RMSE: ' + str(rmse_e_temp))
        print('')
    else:
        pass

ax[0,0].legend(markerscale=3,fontsize='x-large')
ax[0,1].legend(markerscale=3,fontsize='x-large')
ax[1,0].legend(markerscale=3,fontsize='x-large')
ax[1,1].legend(markerscale=3,fontsize='x-large')

fig.savefig('./2bdSOAP/2bd_train_finerRegs.png')



y_pred_train = model_2bd.predict(managers_train_2bd)
f_pred_train = model_2bd.predict_forces(managers_train_2bd)

y_pred = model_2bd.predict(managers_test_2bd)
f_pred = model_2bd.predict_forces(managers_test_2bd)

y_train = y_train - y_pred_train
y_test = y_test - y_pred

f_train = f_train - f_pred_train
f_test = f_test - f_pred


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


### Correlation plots of the training set using 2bd+SOAP models

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12.5,12.5))

ax[0,0].set_xlim(-23,23)
ax[0,0].set_ylim(-23,23)
ax[0,1].set_xlim(-12,12)
ax[0,1].set_ylim(-12,12)
ax[1,0].set_xlim(-6,1)
ax[1,0].set_ylim(-6,1)
ax[1,1].set_xlim(-5.5,-4)
ax[1,1].set_ylim(-5.5,-4)
ax[0,0].plot([0, 1], [0, 1], transform=ax[0,0].transAxes,color='k',linewidth=1)
ax[0,1].plot([0, 1], [0, 1], transform=ax[0,1].transAxes,color='k',linewidth=1)
ax[1,0].plot([0, 1], [0, 1], transform=ax[1,0].transAxes,color='k',linewidth=1)
ax[1,1].plot([0, 1], [0, 1], transform=ax[1,1].transAxes,color='k',linewidth=1)

fig.suptitle('Training set',fontsize=24)
ax[0,0].set_title('GAP-RSS generated data',fontsize=20)
ax[0,0].set_xlabel('Reference forces (eV/A)',fontsize=20)
ax[0,0].set_ylabel('Predicted forces (eV/A)',fontsize=20)
ax[0,0].tick_params(labelsize=18)
ax[0,1].set_title('Manually constructed data',fontsize=20)
ax[0,1].set_xlabel('Reference forces (eV/A)',fontsize=20)
ax[0,1].set_ylabel('Predicted forces (eV/A)',fontsize=20)
ax[0,1].tick_params(labelsize=18)

ax[1,0].set_title('GAP-RSS generated data',fontsize=20)
ax[1,0].set_xlabel('Reference energy (eV/atom)',fontsize=20)
ax[1,0].set_ylabel('Predicted energy (eV/atom)',fontsize=20)
ax[1,0].tick_params(labelsize=18)
ax[1,1].set_title('Manually constructed data',fontsize=20)
ax[1,1].set_xlabel('Reference energy (eV/atom)',fontsize=20)
ax[1,1].set_ylabel('Predicted energy (eV/atom)',fontsize=20)
ax[1,1].tick_params(labelsize=18)

RSS_list = ['rss_rnd', 'rss_005', 'rss_200', 'rss_3c']
manual_list = ['liq_12_03_01_liqP4', 'liq_12_03_02_network', 'phosphorene', 'phosphorus_ribbons','2D','cryst_dist','cryst_dist_hp']

for key in dataset.keys():
    frames_temp = [frames_train[i] for i in dataset[key]]
    y_temp, f_temp, n_temp = extract_ref(frames_temp,True,True,True)
    y_temp_per_atom = y_temp/n_temp
    managers_temp_soap = managers_train.get_subset(dataset[key])
    managers_temp_2bd = managers_train_2bd.get_subset(dataset[key])
    y_pred_temp = model_2bd.predict(managers_temp_2bd) + modelSOAP.predict(managers_temp_soap)
    y_pred_temp_per_atom = y_pred_temp/n_temp
    f_pred_temp = model_2bd.predict_forces(managers_temp_2bd) + modelSOAP.predict_forces(managers_temp_soap)
    
    x = f_temp.flatten()
    y = f_pred_temp.flatten()
    
    if key in RSS_list:
        if key == 'rss_3c':
            order = 10
            color = 'green'
            label = 'RSS, 3-coordinated'
        elif key == 'rss_200':
            order = 8
            color = 'lime'
            label = 'RSS, relaxed'
        elif key == 'rss_005':
            order = 6
            color = 'aqua'
            label = 'RSS, intermediates'
        elif key == 'rss_rnd':
            order = 4
            color = 'dodgerblue'
            label = 'RSS, initial (random)'
        ax[0,0].scatter(x, y, s=6,c=color,marker='o',label=label,zorder=order)
        ax[1,0].scatter(y_temp_per_atom, y_pred_temp_per_atom, s=6,c=color,marker='o',label=label,zorder=order)
        rmse_e_temp = mean_squared_error(y_pred_temp_per_atom, y_temp_per_atom, squared=False)
        rmse_f_temp = mean_squared_error(y, x, squared=False) 
        print(str(key)+':')
        print('Forces RMSE: ' + str(rmse_f_temp))
        print('Energy RMSE: ' + str(rmse_e_temp))
        print('')
    elif key in manual_list:
        if key == 'cryst_dist':
            order = 10
            color = 'green'
            label = 'Bulk crystals'
        elif key == 'phosphorene' or key == 'phosphorus_ribbons':
            order = 8
            color = 'lime'
            label = None
        elif key == '2D':
            order = 8
            color = 'lime'
            label = '2D structures'
        elif key == 'liq_12_03_01_liqP4':
            order = 6
            color = 'aqua'
            label = 'Molecular liquid'
        elif key == 'liq_12_03_02_network':
            order = 4
            color = 'dodgerblue'
            label = 'Network liquid'
        ax[0,1].scatter(x, y, s=6,c=color,marker='o',label=label,zorder=order)
        ax[1,1].scatter(y_temp_per_atom, y_pred_temp_per_atom, s=6,c=color,marker='o',label=label,zorder=order)
        rmse_e_temp = mean_squared_error(y_pred_temp_per_atom, y_temp_per_atom, squared=False)
        rmse_f_temp = mean_squared_error(y, x, squared=False) 
        print(str(key)+':')
        print('Forces RMSE: ' + str(rmse_f_temp))
        print('Energy RMSE: ' + str(rmse_e_temp))
        print('')
    else:
        pass

ax[0,0].legend(markerscale=3,fontsize='x-large')
ax[0,1].legend(markerscale=3,fontsize='x-large')
ax[1,0].legend(markerscale=3,fontsize='x-large')
ax[1,1].legend(markerscale=3,fontsize='x-large')

plt.tight_layout()

fig.savefig('./2bdSOAP/total_train_finerRegs.png')


### Correlation plots of the test set using 2bd+SOAP models

print("")
print("")
print("Test set RMSEs:")
print("")

rmse_e_list = []
rmse_f_list = []
n_e_list = []
n_f_list = []

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12.5,12.5))

ax[0,0].set_xlim(-23,23)
ax[0,0].set_ylim(-23,23)
ax[0,1].set_xlim(-10,10)
ax[0,1].set_ylim(-10,10)
ax[1,0].set_xlim(-6,1)
ax[1,0].set_ylim(-6,1)
ax[1,1].set_xlim(-5.75,-0.5)
ax[1,1].set_ylim(-5.75,-0.5)
ax[0,0].plot([0, 1], [0, 1], transform=ax[0,0].transAxes,color='k',linewidth=1)
ax[0,1].plot([0, 1], [0, 1], transform=ax[0,1].transAxes,color='k',linewidth=1)
ax[1,0].plot([0, 1], [0, 1], transform=ax[1,0].transAxes,color='k',linewidth=1)
ax[1,1].plot([0, 1], [0, 1], transform=ax[1,1].transAxes,color='k',linewidth=1)

fig.suptitle('Test set',fontsize=24)
ax[0,0].set_title('GAP-RSS generated data',fontsize=20)
ax[0,0].set_xlabel('Reference forces (eV/A)',fontsize=20)
ax[0,0].set_ylabel('Predicted forces (eV/A)',fontsize=20)
ax[0,0].tick_params(labelsize=18)
ax[0,1].set_title('Manually constructed data',fontsize=20)
ax[0,1].set_xlabel('Reference forces (eV/A)',fontsize=20)
ax[0,1].set_ylabel('Predicted forces (eV/A)',fontsize=20)
ax[0,1].tick_params(labelsize=18)

ax[1,0].set_title('GAP-RSS generated data',fontsize=20)
ax[1,0].set_xlabel('Reference energy (eV/atom)',fontsize=20)
ax[1,0].set_ylabel('Predicted energy (eV/atom)',fontsize=20)
ax[1,0].tick_params(labelsize=18)
ax[1,1].set_title('Manually constructed data',fontsize=20)
ax[1,1].set_xlabel('Reference energy (eV/atom)',fontsize=20)
ax[1,1].set_ylabel('Predicted energy (eV/atom)',fontsize=20)
ax[1,1].tick_params(labelsize=18)

RSS_list = ['rss_rnd', 'rss_005', 'rss_200', 'rss_3c']
manual_list = ['liq_12_03_01_liqP4', 'liq_12_03_02_network', 'phosphorene', 'phosphorus_ribbons','2D','cryst_dist','cryst_dist_hp']

for key in dataset2.keys():
    frames_temp = [frames_test[i] for i in dataset2[key]]
    y_temp, f_temp, n_temp = extract_ref(frames_temp,True,True,True)
    y_temp_per_atom = y_temp/n_temp
    managers_temp_2bd = soap_2bd.transform(frames_temp)
    managers_temp_soap = soap.transform(frames_temp)
    y_pred_temp = model_2bd.predict(managers_temp_2bd) + modelSOAP.predict(managers_temp_soap)
    y_pred_temp_per_atom = y_pred_temp/n_temp
    f_pred_temp = model_2bd.predict_forces(managers_temp_2bd) + modelSOAP.predict_forces(managers_temp_soap)
    
    x = f_temp.flatten()
    y = f_pred_temp.flatten()
    
    if key in RSS_list:
        if key == 'rss_3c':
            order = 10
            color = 'green'
            label = 'RSS, 3-coordinated'
        elif key == 'rss_200':
            order = 8
            color = 'lime'
            label = 'RSS, relaxed'
        elif key == 'rss_005':
            order = 6
            color = 'aqua'
            label = 'RSS, intermediates'
        elif key == 'rss_rnd':
            order = 4
            color = 'dodgerblue'
            label = 'RSS, initial (random)'
        ax[0,0].scatter(x, y, s=6,c=color,marker='o',label=label,zorder=order)
        ax[1,0].scatter(y_temp_per_atom, y_pred_temp_per_atom, s=6,c=color,marker='o',label=label,zorder=order)
        
        rmse_e_temp = mean_squared_error(y_pred_temp_per_atom, y_temp_per_atom, squared=False)
        rmse_f_temp = mean_squared_error(y, x, squared=False) 
        rmse_e_list.append(rmse_e_temp)
        rmse_f_list.append(rmse_f_temp)
        n_e_list.append(len(y_temp_per_atom))
        n_f_list.append(len(x))
        
        print(str(key)+':')
        print('Forces RMSE: ' + str(rmse_f_temp))
        print('Energy RMSE: ' + str(rmse_e_temp))
        print('')
    elif key in manual_list:
        if key == 'cryst_dist':
            order = 10
            color = 'green'
            label = 'Bulk crystals'
        elif key == 'cryst_dist_hp':
            order = 10
            color = 'black'
            label = 'hp'
        elif key == 'phosphorene' or key == 'phosphorus_ribbons':
            order = 8
            color = 'lime'
            label = None
        elif key == '2D':
            order = 8
            color = 'lime'
            label = '2D structures'
        elif key == 'liq_12_03_01_liqP4':
            order = 6
            color = 'aqua'
            label = 'Molecular liquid'
        elif key == 'liq_12_03_02_network':
            order = 4
            color = 'dodgerblue'
            label = 'Network liquid'
        ax[0,1].scatter(x, y, s=6,c=color,marker='o',label=label,zorder=order)
        ax[1,1].scatter(y_temp_per_atom, y_pred_temp_per_atom, s=6,c=color,marker='o',label=label,zorder=order)
        
        rmse_e_temp = mean_squared_error(y_pred_temp_per_atom, y_temp_per_atom, squared=False)
        rmse_f_temp = mean_squared_error(y, x, squared=False)
        rmse_e_list.append(rmse_e_temp)
        rmse_f_list.append(rmse_f_temp)
        n_e_list.append(len(y_temp_per_atom))
        n_f_list.append(len(x))
        
        print(str(key)+':')
        print('Forces RMSE: ' + str(rmse_f_temp))
        print('Energy RMSE: ' + str(rmse_e_temp))
        print('')
    else:
        pass

ax[0,0].legend(markerscale=3,fontsize='x-large')
ax[0,1].legend(markerscale=3,fontsize='x-large')
ax[1,0].legend(markerscale=3,fontsize='x-large')
ax[1,1].legend(markerscale=3,fontsize='x-large')

plt.tight_layout()

fig.savefig('./2bdSOAP/total_test_finerRegs.png')

n_e_total = sum(n_e_list)
n_f_total = sum(n_f_list)
rmse_e_total = 0
rmse_f_total = 0

for i in np.arange(len(n_e_list)):
    rmse_e_total += n_e_list[i]*np.power(rmse_e_list[i],2)

rmse_e_total = np.sqrt(rmse_e_total/n_e_total)

for i in np.arange(len(n_f_list)):
    rmse_f_total += n_f_list[i]*np.power(rmse_f_list[i],2)

rmse_f_total = np.sqrt(rmse_f_total/n_f_total)

print("Total RMSEs of the test set:")
print("Forces RMSE: "+ str(rmse_f_total))
print("Energy RMSE: "+ str(rmse_e_total))

end = time.time()

print('')
print("The time of execution is :", end-start)
