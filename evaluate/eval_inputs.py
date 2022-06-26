import numpy as np
np.seterr(all="ignore")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_rgb, hex2color
import mplhep as hep
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
#from torch.utils.data import TensorDataset, ConcatDataset

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import entropy

import gc

import coffea.hist as hist

import time

import argparse
import ast


import sys
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/helpers/")
from tools import preprocessed_path, eval_path, FALLBACK_NUM_DATASETS, get_all_defaults, get_all_scalers, build_wm_text_dict
from variables import input_indices_wanted, integer_indices, full_index_to_name, full_index_to_text, full_index_to_unit, full_index_to_digit, full_index_to_range, get_full_index_from_non_target_index, get_wanted_full_indices

sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/attack/")
from disturb_inputs import fgsm_attack, apply_noise
# ToDo
#import definitions
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/training/")
from focal_loss import FocalLoss, focal_loss
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])

# for reproducible results
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("variable", type=int, help="Index of input variable")
parser.add_argument("attack", help="The type of the attack, noise or fgsm")
parser.add_argument("para", help="Parameter for attack or noise (epsilon or sigma), can be comma-separated.")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", help="Number of previously trained epochs, can be a comma-separated list")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _altptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible")
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("restrict", help="Restrict impact of the attack ? -1 for no, some positive value for yes")
args = parser.parse_args()

variable = args.variable
attack = args.attack
#fixRange = args.fixRange
para = args.para
param = [float(p) for p in para.split(',')]
NUM_DATASETS = args.files
NUM_DATASETS = FALLBACK_NUM_DATASETS if NUM_DATASETS < 0 else NUM_DATASETS
at_epoch = args.prevep
epochs = [int(e) for e in at_epoch.split(',')]
weighting_method = args.wm
wmets = [w for w in weighting_method.split(',')]
    
n_samples = args.jets
    
restrict_impact = args.restrict

print(f'Evaluate training at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')

gamma = [((weighting_method.split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0] for weighting_method in wmets]
alphaparse = [((weighting_method.split('_gamma')[-1]).split('_alpha')[-1]).split('_adv_tr_eps')[0] for weighting_method in wmets]
epsilon = [(weighting_method.split('_adv_tr_eps')[-1]) for weighting_method in wmets]
print('gamma',gamma)
print('alpha',alphaparse)
print('epsilon',epsilon)

colorcode = ['#B22222','#00FFFF','#006400']
opacity_raw = 0.42

wm_def_text = build_wm_text_dict(gamma,alphaparse,epsilon)



test_input_file_paths = [preprocessed_path + f'test_inputs_%d.pt' % k for k in range(NUM_DATASETS)]
test_target_file_paths = [preprocessed_path + f'test_targets_%d.pt' % k for k in range(NUM_DATASETS)]
    
    
    
# note: the default case without parameters in the function input_indices_wanted return all high-level variables, as well as 28 (all) features for the first 6 tracks
used_variables = input_indices_wanted()
slices = torch.LongTensor(used_variables)
# use n_input_features as the number of inputs to the model (later)
n_input_features = len(slices)
    
    
    
scalers = get_all_scalers()


test_targets = torch.cat(tuple(torch.load(ti).to(device) for ti in test_target_file_paths))

relative_entropies = []


plt.ioff()
model = nn.Sequential(nn.Linear(n_input_features, 100),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(100, 100),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(100, 100),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(100, 100),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(100, 100),
                                  nn.ReLU(),
                                  nn.Linear(100, 3),
                                  nn.Softmax(dim=1))
    
model.to(device)
model.eval()
    

n_compare = 1
checkpoint = torch.load(f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{wmets[0]}_{NUM_DATASETS}_{n_samples}/model_{epochs[0]}_epochs{wmets[0]}_{NUM_DATASETS}_datasets_{n_samples}.pt', map_location=torch.device(device))
model.load_state_dict(checkpoint["model_state_dict"])
if ('_ptetaflavloss' in wmets[0]) or ('_altptetaflavloss' in wmets[0]):
    if 'focalloss' not in wmets[0]:
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif 'focalloss' in wmets[0]:
        if 'alpha' not in wmets[0]:
            alpha = None
        else:
            commasep_alpha = [a for a in ((wmets[0].split('_alpha')[-1]).split('_adv_tr_eps')[0]).split(',')]
            alpha = torch.Tensor([float(commasep_alpha[0]),float(commasep_alpha[1]),float(commasep_alpha[2])]).to(device)
        if 'gamma' not in wmets[0]:
            gamma = 2.0
        else:
            gamma = float(((wmets[0].split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0])
        criterion = FocalLoss(alpha, gamma, reduction='none')

else:
    criterion = nn.CrossEntropyLoss()
        
print('Loaded model and corresponding criterion.')
        
def plot(variable=0,mode=attack,param=0.1,minim=None,maxim=None,reduced=True,restrict=float(restrict_impact)):
    # variable will be the used variable (if one only takes six tracks into account)
    # transform to full variable by doing
    non_target_variable = used_variables[variable]
    full_variable = get_full_index_from_non_target_index(non_target_variable)
    # full variable will stay no matter what choice one makes,
    # if one would use the non-full variables, one might overwrite plots later if a
    # different choice is made to the selected indices
    print('full_variable:',full_variable)
    
    short_name = full_index_to_name(full_variable)
    display_name = full_index_to_text(full_variable)
    print(f'Doing {short_name} = {display_name} now.')
    unit = full_index_to_unit(full_variable)
    #unit = ''
    xunit = '' if unit == '' else f' [{unit}]'
    xmagn = []
    if minim == None and maxim == None:
        min_max = full_index_to_range(full_variable)
        minim = min_max[0]
        maxim = min_max[1]
    
    restrict_text = f'_restrictedBy{restrict}' if restrict > 0 else '_restrictedByInf'
    print('perturbation',restrict_text)
    
    # to test with one file only for quick checks
    #for s in range(0, 1):
    for s in range(0, len(test_target_file_paths)):
        #scalers = torch.load(scalers_file_paths[s])
        #scalers = all_scalers[s]
        #test_inputs =  torch.load(test_input_file_paths[s]).to(device).float()
        all_inputs =  (torch.load(test_input_file_paths[s])[:,slices]).to(device)
        #all_inputs =  (torch.load(test_input_file_paths[s])[:,slices]).to(device)[:10]
        #val_inputs =  torch.load(val_input_file_paths[s]).to(device).float()
        #train_inputs =  torch.load(train_input_file_paths[s]).to(device).float()
        #test_targets =  torch.load(test_target_file_paths[s]).to(device)
        all_targets =  torch.load(test_target_file_paths[s]).to(device)
        #all_targets =  torch.load(test_target_file_paths[s]).to(device)[:10]
        
        # if you want to test on one file only (replaces the total test_target variable with the current all_targets variable)
        #global test_targets
        #test_targets = all_targets
        
        #val_targets =  torch.load(val_target_file_paths[s]).to(device)
        #train_targets =  torch.load(train_target_file_paths[s]).to(device)
        #all_inputs = torch.cat((test_inputs,val_inputs,train_inputs))
        #all_targets = torch.cat((test_targets,val_targets,train_targets))
        
        integervars = integer_indices
        
        for i, m in enumerate(param):
            if s > 0:
                if mode == 'fgsm':
                    xadv = np.concatenate((xmagn[i], scalers[full_variable].inverse_transform(fgsm_attack(epsilon=param[i],sample=all_inputs,targets=all_targets,thismodel=model,thiscriterion=criterion,filtered_indices=used_variables,restrict_impact=restrict)[:,variable].cpu())))
                else:
                    xadv = np.concatenate((xmagn[i], scalers[full_variable].inverse_transform(apply_noise(all_inputs,magn=param[i],filtered_indices=used_variables,restrict_impact=restrict)[:,variable].cpu())))
                if full_variable in integervars:
                    xadv = np.rint(xadv)
                xmagn[i] = xadv
            else:
                if mode == 'fgsm':
                    xadv = scalers[full_variable].inverse_transform(fgsm_attack(epsilon=param[i],sample=all_inputs,targets=all_targets,thismodel=model,thiscriterion=criterion,filtered_indices=used_variables,restrict_impact=restrict)[:,variable].cpu())
                else:
                    xadv = scalers[full_variable].inverse_transform(apply_noise(all_inputs,magn=param[i],filtered_indices=used_variables,restrict_impact=restrict)[:,variable].cpu())
                if full_variable in integervars:
                    xadv = np.rint(xadv)
                    
                xmagn.append(xadv)
        
        
        del all_inputs
        del all_targets
        gc.collect()
    
     
    minimum = min([min(xmagn[i]) for i in range(len(param))])-0.01
    maximum = max([max(xmagn[i]) for i in range(len(param))])+0.01
    print(minimum, maximum)
    
    if full_variable not in integervars:
        # bins are actually bin centers (to plot the errorbars in the ratio plot)
        bins = np.linspace(minimum+(maximum-minimum)/50/2,maximum-(maximum-minimum)/50/2,50)

        compHist = hist.Hist("Jets",
                              hist.Cat("sample","sample name"),
                              hist.Bin("prop",display_name+xunit,50,minimum,maximum))
        newHist = hist.Hist("Jets",
                              hist.Cat("sample","sample name"),
                              hist.Cat("flavour","flavour of the jet"),
                              hist.Bin("prop",display_name+xunit,50,minimum,maximum))
    
    elif full_variable in integervars:
        #bins = np.arange(0,maximum+1)
        bin_edges = np.arange(int(minimum),int(maximum)+2)-0.5
        bins = (bin_edges[0:-1]+bin_edges[1:])/2
        compHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Bin("prop",display_name+unit,bin_edges))
        newHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Cat("flavour","flavour of the jet"),
                          hist.Bin("prop",display_name+unit,bin_edges))
    
    bin_size_original = bins[2] - bins[1]
    bin_size_reduced = round(bin_size_original,full_index_to_digit(full_variable))
    
    compHist.fill(sample="raw",prop=xmagn[0])
    newHist.fill(sample="raw",flavour='b-jets',prop=xmagn[0][test_targets == 0])
    newHist.fill(sample="raw",flavour='c-jets',prop=xmagn[0][test_targets == 1])
    newHist.fill(sample="raw",flavour='udsg-jets',prop=xmagn[0][test_targets == 2])
    
    for si in range(1,len(param)):
        if mode == 'fgsm':
            compHist.fill(sample=f"fgsm $\epsilon$={param[si]}",prop=xmagn[si])
            newHist.fill(sample=f"fgsm $\epsilon$={param[si]}",flavour='b-jets',prop=xmagn[si][test_targets == 0])
            newHist.fill(sample=f"fgsm $\epsilon$={param[si]}",flavour='c-jets',prop=xmagn[si][test_targets == 1])
            newHist.fill(sample=f"fgsm $\epsilon$={param[si]}",flavour='udsg-jets',prop=xmagn[si][test_targets == 2])
        else:
            compHist.fill(sample=f"noise $\sigma$={param[si]}",prop=xmagn[si])
            newHist.fill(sample=f"noise $\sigma$={param[si]}",flavour='b-jets',prop=xmagn[si][test_targets == 0])
            newHist.fill(sample=f"noise $\sigma$={param[si]}",flavour='c-jets',prop=xmagn[si][test_targets == 1])
            newHist.fill(sample=f"noise $\sigma$={param[si]}",flavour='udsg-jets',prop=xmagn[si][test_targets == 2])
            
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2],'hspace': .25})
    if len(param) == 3:
        hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    elif len(param) == 2:
        hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#1f77b4']})
    ax1.get_legend().remove()
    
    
    if full_variable in integervars:
        ax1.set_ylabel('Jets'+' / '+'1 unit', fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit, fontsize=21)
    
    if mode == 'fgsm':
        if len(param) == 3:
            ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'],fontsize=15)
        if len(param) == 2:
            ax1.legend([f'FGSM $\epsilon$={param[1]}','Raw'],fontsize=15)
    else:
        if len(param) == 3:
            ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'],fontsize=15)
        if len(param) == 2:
            ax1.legend([f'Noise $\sigma$={param[1]}','Raw'],fontsize=15)
        
    running_relative_entropies = []
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        '''
            Kullback-Leibler divergence with raw (denom ratio plot) and disturbed (num ratio plot) data: relative entropy
            
            As explained above
        '''
        num[(num == 0) & (denom != 0)] = 1
        entr = entropy(denom, qk=num)
        running_relative_entropies.append([full_variable, param[si], entr])
        
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.set_ylim(0,2)
        if full_variable not in integervars:
            ax2.plot([minimum,maximum],[1,1],color='black')    
            ax2.set_xlim(minimum,maximum)
        elif full_variable in integervars:
            if len(bins) < 20:
                ax2.set_xticks(bins)
            ax2.plot([minimum-0.5,maximum+0.5],[1,1],color='black')  
            ax2.set_xlim(minimum-0.5,maximum+0.5)
        
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw',loc='center', fontsize=21)
        else:
            ax2.set_ylabel('Noise/raw',loc='center', fontsize=21)
            
            
    relative_entropies.append(running_relative_entropies)
    print(relative_entropies)
    
    
    range_text = ''
    log_text = ''
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    
    kl = np.array(relative_entropies)
    print(kl)
    np.save(eval_path+f'inputs/kl_div/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}.npy', kl)
    plt.show(block=False)
    time.sleep(5)
    plt.close('all')
    del fig, ax1, ax2
    gc.collect(2)
    
    
    # !!! logarithmic axis --> separate plot
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2],'hspace': 0.0})
    ax1 = hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.set_yscale('log')
    #ax1.set_ylim(None, None)
    ax1.set_ylim(bottom=None)
    ax1.relim()
    ax1.autoscale_view()
    ax1.autoscale()
    ax1.get_legend().remove()
    
    
    if full_variable in integervars:
        ax1.set_ylabel('Jets'+' / '+'1 unit', fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit, fontsize=21)
            
    
    if mode == 'fgsm':
        ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'],fontsize=15)
    else:
        ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'],fontsize=15)
        
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.set_ylim(0,2)
        if full_variable not in integervars:
            ax2.plot([minimum,maximum],[1,1],color='black')    
            ax2.set_xlim(minimum,maximum)
        elif full_variable in integervars:
            if len(bins) < 20:
                ax2.set_xticks(bins)
            ax2.plot([minimum-0.5,maximum+0.5],[1,1],color='black')  
            ax2.set_xlim(minimum-0.5,maximum+0.5)
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw',loc='center', fontsize=21)
        else:
            ax2.set_ylabel('Noise/raw',loc='center', fontsize=21)
            
    xlbl = ax1.get_xlabel()
    ax2.set_xlabel(xlbl)        
    range_text = ''
    log_text = '_logAxis'
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    del fig, ax1, ax2
    gc.collect()
    
    # stop for debugging of first part
    #sys.exit()

    # ===============================================================================================================
    #
    #
    #                                        Split input shapes by flavour
    #
    #
    # ---------------------------------------------------------------------------------------------------------------
    
    
    
    # old version with ratio plot --> not used
    fig, ax1 = plt.subplots(1,1,figsize=[10,6])
    hist.plot1d(newHist['raw'].sum('sample'),overlay='flavour',ax=ax1,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        
        
    for si in range(2,len(param)):
        ax1.set_prop_cycle(None)
        if mode == 'fgsm':
            hist.plot1d(newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
            numB  = newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample')['b-jets'].sum('flavour').values()[()]
            numC  = newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample')['c-jets'].sum('flavour').values()[()]
            numL  = newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample')['udsg-jets'].sum('flavour').values()[()]
        else:
            hist.plot1d(newHist[f"noise $\sigma$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            numB  = newHist[f"noise $\sigma$={param[si]}"].sum('sample')['b-jets'].sum('flavour').values()[()]
            numC  = newHist[f"noise $\sigma$={param[si]}"].sum('sample')['c-jets'].sum('flavour').values()[()]
            numL  = newHist[f"noise $\sigma$={param[si]}"].sum('sample')['udsg-jets'].sum('flavour').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        denomB  = newHist['raw'].sum('sample')['b-jets'].sum('flavour').values()[()]
        denomC  = newHist['raw'].sum('sample')['c-jets'].sum('flavour').values()[()]
        denomL  = newHist['raw'].sum('sample')['udsg-jets'].sum('flavour').values()[()]
        ratio = num / denom
        ratioB  = numB / denomB
        ratioC  = numC / denomC
        ratioL  = numL / denomL
        num_err = np.sqrt(num)
        num_errB  = np.sqrt(numB )
        num_errC  = np.sqrt(numC )
        num_errL  = np.sqrt(numL )
        denom_err = np.sqrt(denom)
        denom_errB  = np.sqrt(denomB )
        denom_errC  = np.sqrt(denomC )
        denom_errL  = np.sqrt(denomL )
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        ratio_errB  = np.sqrt((num_errB/denomB)**2+(numB/(denomB**2)*denom_errB)**2)
        ratio_errC  = np.sqrt((num_errC/denomC)**2+(numC/(denomC**2)*denom_errC)**2)
        ratio_errL  = np.sqrt((num_errL/denomL)**2+(numL/(denomL**2)*denom_errL)**2)
    
    if full_variable in integervars:
        if len(bins) < 20:
            ax1.set_xticks(bins)
        ax1.set_xlim(minimum-0.5,maximum+0.5)
        ax1.set_ylabel('Jets'+' / '+'1 unit', fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit, fontsize=21)
            
    
    ax1.get_legend().remove()      
    if mode == 'fgsm':
        legend_labels = [f'b ($\epsilon$ = {param[si]})',f'c ($\epsilon$ = {param[si]})',f'udsg ($\epsilon$ = {param[si]})','b (raw)','c (raw)','udsg (raw)']
        ax1.legend(legend_labels,fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    else:
        legend_labels = [f'b ($\sigma$ = {param[si]})',f'c ($\sigma$ = {param[si]})',f'udsg ($\sigma$ = {param[si]})','b (raw)','c (raw)','udsg (raw)']
        ax1.legend(legend_labels,fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    
    handles, labels = ax1.get_legend_handles_labels()
    handles = np.concatenate((handles[::3],handles[1::3],handles[2::3]),axis=0)
    labels = np.concatenate((legend_labels[::3],legend_labels[1::3],legend_labels[2::3]),axis=0)
    ax1.legend(handles, labels,fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left', mode="expand")
    
    range_text = ''
    log_text = ''
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    
    
    # old version with ratio plot --> not used
    #del fig, ax1, ax2
    del fig, ax1
    gc.collect(2)
    
    
    # !!! logarithmic axis --> separate plot
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2],'hspace': .25})
    fig, ax1 = plt.subplots(1,1,figsize=[10,6])
    hist.plot1d(newHist['raw'].sum('sample'),overlay='flavour',ax=ax1,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        
    for si in range(2,len(param)):
        ax1.set_prop_cycle(None)
        if mode == 'fgsm':
            hist.plot1d(newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
        else:
            hist.plot1d(newHist[f"noise $\sigma$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
    
    ax1.set_yscale('log')
    #ax1.set_ylim(None, None)
    ax1.set_ylim(bottom=None)
    ax1.relim()
    ax1.autoscale_view()
    ax1.autoscale()        
    
    
    if full_variable in integervars:
        if len(bins) < 20:
            ax1.set_xticks(bins)
        ax1.set_xlim(minimum-0.5,maximum+0.5)
        ax1.set_ylabel('Jets'+' / '+'1 unit', fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit, fontsize=21)
    
    ax1.get_legend().remove()      
    if mode == 'fgsm':
        ax1.legend([f'b ($\epsilon$ = {param[si]})',f'c ($\epsilon$ = {param[si]})',f'udsg ($\epsilon$ = {param[si]})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    else:
        ax1.legend([f'b ($\sigma$ = {param[si]})',f'c ($\sigma$ = {param[si]})',f'udsg ($\sigma$ = {param[si]})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    
    handles, labels = ax1.get_legend_handles_labels()
    handles = np.concatenate((handles[::3],handles[1::3],handles[2::3]),axis=0)
    labels = np.concatenate((legend_labels[::3],legend_labels[1::3],legend_labels[2::3]),axis=0)
    ax1.legend(handles, labels,fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left', mode="expand")
    
    range_text = ''
    log_text = '_logAxis'
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    # old version with ratio plot --> not used
    #del fig, ax1, ax2
    del fig, ax1
    gc.collect()
    
    
    
    # =================================================================================================================
    # 
    #
    #                                                 Fixed range!
    #
    #
    # -----------------------------------------------------------------------------------------------------------------
    
    
    if minim is None:
        minimum = min([min(xmagn[i]) for i in range(len(param))])-0.01
    else:
        minimum = minim
    if maxim is None:
        maximum = max([max(xmagn[i]) for i in range(len(param))])+0.01
    else:
        maximum = maxim
    
    print(minimum, maximum)
          
    
    if full_variable not in integervars:
        # bins are actually bin centers (to plot the errorbars in the ratio plot)
        bins = np.linspace(minimum+(maximum-minimum)/50/2,maximum-(maximum-minimum)/50/2,50)

        compHist = hist.Hist("Jets",
                              hist.Cat("sample","sample name"),
                              hist.Bin("prop",display_name+xunit,50,minimum,maximum))
        newHist = hist.Hist("Jets",
                              hist.Cat("sample","sample name"),
                              hist.Cat("flavour","flavour of the jet"),
                              hist.Bin("prop",display_name+xunit,50,minimum,maximum))
    
    elif full_variable in integervars:
        bin_edges = np.arange(int(minimum),int(maximum)+2)-0.5
        bins = (bin_edges[0:-1]+bin_edges[1:])/2
        compHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Bin("prop",display_name+xunit,bin_edges))
        newHist = hist.Hist("Jets",
                          hist.Cat("sample","sample name"),
                          hist.Cat("flavour","flavour of the jet"),
                          hist.Bin("prop",display_name+xunit,bin_edges))
    
    bin_size_original = bins[2] - bins[1]
    bin_size_reduced = round(bin_size_original,full_index_to_digit(full_variable))
    
    compHist.fill(sample="raw",prop=xmagn[0])
    newHist.fill(sample="raw",flavour='b-jets',prop=xmagn[0][test_targets == 0])
    newHist.fill(sample="raw",flavour='c-jets',prop=xmagn[0][test_targets == 1])
    newHist.fill(sample="raw",flavour='udsg-jets',prop=xmagn[0][test_targets == 2])
    
    for si in range(1,len(param)):
        if mode == 'fgsm':
            compHist.fill(sample=f"fgsm $\epsilon$={param[si]}",prop=xmagn[si])
            newHist.fill(sample=f"fgsm $\epsilon$={param[si]}",flavour='b-jets',prop=xmagn[si][test_targets == 0])
            newHist.fill(sample=f"fgsm $\epsilon$={param[si]}",flavour='c-jets',prop=xmagn[si][test_targets == 1])
            newHist.fill(sample=f"fgsm $\epsilon$={param[si]}",flavour='udsg-jets',prop=xmagn[si][test_targets == 2])
        else:
            compHist.fill(sample=f"noise $\sigma$={param[si]}",prop=xmagn[si])
            newHist.fill(sample=f"noise $\sigma$={param[si]}",flavour='b-jets',prop=xmagn[si][test_targets == 0])
            newHist.fill(sample=f"noise $\sigma$={param[si]}",flavour='c-jets',prop=xmagn[si][test_targets == 1])
            newHist.fill(sample=f"noise $\sigma$={param[si]}",flavour='udsg-jets',prop=xmagn[si][test_targets == 2])
            
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2],'hspace': .25})
    hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    
    
    if full_variable in integervars:
        ax1.set_ylabel('Jets'+' / '+'1 unit',fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit, fontsize=21)
    
    ax1.get_legend().remove()
    if mode == 'fgsm':
        ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'],fontsize=15)
    else:
        ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'],fontsize=15)
        
    running_relative_entropies = []
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        '''
            Kullback-Leibler divergence with raw (denom ratio plot) and disturbed (num ratio plot) data: relative entropy
            
            As explained above
        '''
        num[(num == 0) & (denom != 0)] = 1
        entr = entropy(denom, qk=num)
        running_relative_entropies.append([variable, param[si], entr])
        
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.set_ylim(0,2)
        if full_variable not in integervars:
            ax2.plot([minimum,maximum],[1,1],color='black')    
            ax2.set_xlim(minimum,maximum)
        elif full_variable in integervars:
            if len(bins) < 20:
                ax2.set_xticks(bins)
            ax2.plot([minimum-0.5,maximum+0.5],[1,1],color='black')  
            ax2.set_xlim(minimum-0.5,maximum+0.5)
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw',loc='center', fontsize=21)
        else:
            ax2.set_ylabel('Noise/raw',loc='center', fontsize=21)
            
            
    relative_entropies.append(running_relative_entropies)
    print(relative_entropies)
        
        
    range_text = '_specRange'
    log_text = ''
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    
    kl = np.array(relative_entropies)
    print(kl)
    np.save(eval_path+f'inputs/kl_div/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}.npy', kl)
    del fig, ax1, ax2
    gc.collect(2)
    
    
    # !!! logarithmic axis --> separate plot
    
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2],'hspace': 0.0})
    ax1 = hist.plot1d(compHist,overlay='sample',ax=ax1,line_opts={'c':['#ff7f0e', '#2ca02c', '#1f77b4']})
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=None)
    ax1.relim()
    ax1.autoscale_view()
    ax1.autoscale()
    
    
    
    if full_variable in integervars:
        ax1.set_ylabel('Jets'+' / '+'1 unit', fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit, fontsize=21)
    
    ax1.get_legend().remove()
    if mode == 'fgsm':
        ax1.legend([f'FGSM $\epsilon$={param[1]}',f'FGSM $\epsilon$={param[2]}','Raw'],fontsize=15)
    else:
        ax1.legend([f'Noise $\sigma$={param[1]}',f'Noise $\sigma$={param[2]}','Raw'],fontsize=15)
        
    for si in range(1,len(param)):
        if mode == 'fgsm':
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
        else:
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        ratio = num / denom
        num_err = np.sqrt(num)
        denom_err = np.sqrt(denom)
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        
        if si == 1:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#ff7f0e')
        else:
            ax2.errorbar(bins,ratio,yerr=ratio_err,fmt='.',color='#2ca02c')
        ax2.set_ylim(0,2)
        if full_variable not in integervars:
            ax2.plot([minimum,maximum],[1,1],color='black')    
            ax2.set_xlim(minimum,maximum)
        elif full_variable in integervars:
            if len(bins) < 20:
                ax2.set_xticks(bins)
            ax2.plot([minimum-0.5,maximum+0.5],[1,1],color='black')  
            ax2.set_xlim(minimum-0.5,maximum+0.5)
        if mode == 'fgsm':
            ax2.set_ylabel('FGSM/raw',loc='center', fontsize=21)
        else:
            ax2.set_ylabel('Noise/raw',loc='center', fontsize=21)
            
    
    xlbl = ax1.get_xlabel()
    ax2.set_xlabel(xlbl)
    #ax1.xaxis.get_label().remove()
    range_text = '_specRange'
    log_text = '_logAxis'
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/input_{full_variable}_{short_name}_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    del fig, ax1, ax2
    gc.collect()
    
    # ===============================================================================================================
    #
    #
    #                                        Split input shapes by flavour
    #
    #
    # ---------------------------------------------------------------------------------------------------------------
    
    fig, ax1 = plt.subplots(1,1,figsize=[10,6])
    hist.plot1d(newHist['raw'].sum('sample'),overlay='flavour',ax=ax1,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        
        
    for si in range(2,len(param)):
        ax1.set_prop_cycle(None)
        if mode == 'fgsm':
            hist.plot1d(newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
            num = compHist[f"fgsm $\epsilon$={param[si]}"].sum('sample').values()[()]
            numB  = newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample')['b-jets'].sum('flavour').values()[()]
            numC  = newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample')['c-jets'].sum('flavour').values()[()]
            numL  = newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample')['udsg-jets'].sum('flavour').values()[()]
        else:
            hist.plot1d(newHist[f"noise $\sigma$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
            num = compHist[f"noise $\sigma$={param[si]}"].sum('sample').values()[()]
            numB  = newHist[f"noise $\sigma$={param[si]}"].sum('sample')['b-jets'].sum('flavour').values()[()]
            numC  = newHist[f"noise $\sigma$={param[si]}"].sum('sample')['c-jets'].sum('flavour').values()[()]
            numL  = newHist[f"noise $\sigma$={param[si]}"].sum('sample')['udsg-jets'].sum('flavour').values()[()]
            
        denom = compHist['raw'].sum('sample').values()[()]
        denomB  = newHist['raw'].sum('sample')['b-jets'].sum('flavour').values()[()]
        denomC  = newHist['raw'].sum('sample')['c-jets'].sum('flavour').values()[()]
        denomL  = newHist['raw'].sum('sample')['udsg-jets'].sum('flavour').values()[()]
        ratio = num / denom
        ratioB  = numB / denomB
        ratioC  = numC / denomC
        ratioL  = numL / denomL
        num_err = np.sqrt(num)
        num_errB  = np.sqrt(numB )
        num_errC  = np.sqrt(numC )
        num_errL  = np.sqrt(numL )
        denom_err = np.sqrt(denom)
        denom_errB  = np.sqrt(denomB )
        denom_errC  = np.sqrt(denomC )
        denom_errL  = np.sqrt(denomL )
        ratio_err = np.sqrt((num_err/denom)**2+(num/(denom**2)*denom_err)**2)
        ratio_errB  = np.sqrt((num_errB/denomB)**2+(numB/(denomB**2)*denom_errB)**2)
        ratio_errC  = np.sqrt((num_errC/denomC)**2+(numC/(denomC**2)*denom_errC)**2)
        ratio_errL  = np.sqrt((num_errL/denomL)**2+(numL/(denomL**2)*denom_errL)**2)
        
    
    if full_variable in integervars:
        if len(bins) < 20:
            ax1.set_xticks(bins)
        ax1.set_xlim(minimum-0.5,maximum+0.5)
        ax1.set_ylabel('Jets'+' / '+'1 unit',loc='center', fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit, fontsize=21)
    
    ax1.get_legend().remove()      
    if mode == 'fgsm':
        ax1.legend([f'b ($\epsilon$ = {param[si]})',f'c ($\epsilon$ = {param[si]})',f'udsg ($\epsilon$ = {param[si]})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    else:
        ax1.legend([f'b ($\sigma$ = {param[si]})',f'c ($\sigma$ = {param[si]})',f'udsg ($\sigma$ = {param[si]})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    
    handles, labels = ax1.get_legend_handles_labels()
    handles = np.concatenate((handles[::3],handles[1::3],handles[2::3]),axis=0)
    labels = np.concatenate((legend_labels[::3],legend_labels[1::3],legend_labels[2::3]),axis=0)
    ax1.legend(handles, labels,fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left', mode="expand")
    
    range_text = '_specRange'
    log_text = ''
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    
    
    # old version with ratio plot --> not used
    #del fig, ax1, ax2
    del fig, ax1
    gc.collect(2)
    
    
    # !!! logarithmic axis --> separate plot
    
    # old version with ratio plot --> not used
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=[10,6],gridspec_kw={'height_ratios': [3, 2],'hspace': .25})
    fig, ax1 = plt.subplots(1,1,figsize=[10,6])
    hist.plot1d(newHist['raw'].sum('sample'),overlay='flavour',ax=ax1,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        
    for si in range(2,len(param)):
        ax1.set_prop_cycle(None)
        if mode == 'fgsm':
            hist.plot1d(newHist[f"fgsm $\epsilon$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
        else:
            hist.plot1d(newHist[f"noise $\sigma$={param[si]}"].sum('sample'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linestyle':'-','linewidth':2})
    
    ax1.set_yscale('log')
    #ax1.set_ylim(None, None)
    ax1.set_ylim(bottom=None)
    ax1.relim()
    ax1.autoscale_view()
    ax1.autoscale()    
    
    
    if full_variable in integervars:
        if len(bins) < 20:
            ax1.set_xticks(bins)
        ax1.set_xlim(minimum-0.5,maximum+0.5)
        ax1.set_ylabel('Jets'+' / '+'1 unit', fontsize=21)
        
    else:
        if unit == '':
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+'units', fontsize=21)
        else:
            ax1.set_ylabel('Jets'+' / '+str(bin_size_reduced)+' '+unit,fontsize=21)
    
    ax1.get_legend().remove()    
    if mode == 'fgsm':
        ax1.legend([f'b ($\epsilon$ = {param[si]})',f'c ($\epsilon$ = {param[si]})',f'udsg ($\epsilon$ = {param[si]})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=3,mode='expand',bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    else:
        ax1.legend([f'b ($\sigma$ = {param[si]})',f'c ($\sigma$ = {param[si]})',f'udsg ($\sigma$ = {param[si]})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left')
    
    handles, labels = ax1.get_legend_handles_labels()
    handles = np.concatenate((handles[::3],handles[1::3],handles[2::3]),axis=0)
    labels = np.concatenate((legend_labels[::3],legend_labels[1::3],legend_labels[2::3]),axis=0)
    ax1.legend(handles, labels,fontsize=12,ncol=3,bbox_to_anchor=(0, 1.02,1,0.2),loc='lower left', mode="expand")
    
    range_text = '_specRange'
    log_text = '_logAxis'
    n_samples_text = int(sum(denom))
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.svg', bbox_inches='tight')
    fig.savefig(eval_path+f'inputs/split_by_flav/input_{full_variable}_{short_name}_splitbyflav_{attack}{param}{range_text}{log_text}_{n_samples_text}_{weighting_method}_{at_epoch}{restrict_text}_v2.pdf', bbox_inches='tight')
    
    # old version with ratio plot --> not used
    #del fig, ax1, ax2
    del fig, ax1
    gc.collect()
    
min_max = [None,None]

plot(variable,mode=attack,param=[0]+param,minim=min_max[0],maxim=min_max[1],reduced=True)
