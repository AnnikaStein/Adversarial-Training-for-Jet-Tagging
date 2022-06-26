import numpy as np
import torch

import sys

sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/helpers/")
from tools import defaults_path, preprocessed_path, get_all_scalers, get_all_defaults
from variables import integer_indices, n_input_features, get_wanted_full_indices, all_factor_epsilons

all_scalers = np.array(get_all_scalers())
all_defaults_scaled = np.array(get_all_defaults(scaled=True))
all_defaults = np.array(get_all_defaults(scaled=False))

def apply_noise(sample, magn=1e-2,offset=[0], dev="cpu", filtered_indices=[i for i in range(n_input_features)],restrict_impact=-1):
    seed = 0
    np.random.seed(seed)
        
    if magn == 0:
        return sample
    n_Vars = len(filtered_indices)
    
    wanted_full_indices = get_wanted_full_indices(filtered_indices)
    
    scalers = all_scalers[wanted_full_indices]
    
    defaults_per_variable = all_defaults[wanted_full_indices]
    scaled_defaults_per_variable = all_defaults_scaled[wanted_full_indices]
    
    device = torch.device(dev)

    with torch.no_grad():
        noise = torch.Tensor(np.random.normal(offset,magn,(len(sample),n_Vars))).to(device)
        xadv = sample + noise
        
        # use full indices and check if in int.vars. or defaults
        for i in range(n_Vars):
            if wanted_full_indices[i] in integer_indices:
                xadv[:,i] = sample[:,i]
            else: # non integer, but might have defaults that should be excluded from shift
                defaults = sample[:,i].cpu() == scaled_defaults_per_variable[i]
                if torch.sum(defaults) != 0:
                    xadv[:,i][defaults] = sample[:,i][defaults]
                    
                if restrict_impact > 0:
                    original_back = scalers[i].inverse_transform(sample[:,i])
                    difference_back = scalers[i].inverse_transform(xadv[:,i]) - original_back
                    allowed_perturbation = restrict_impact * np.abs(original_back)
                    high_impact = np.abs(difference_back) > allowed_perturbation
                    if np.sum(high_impact)!=0:
                        scaled_back_max_perturbed = torch.from_numpy(original_back[high_impact]) + torch.from_numpy(allowed_perturbation[high_impact]) * torch.sign(noise[high_impact,i])
                        xadv[high_impact,i] = torch.Tensor(scalers[i].transform(scaled_back_max_perturbed.reshape(-1,1)).flatten())

        return xadv

def fgsm_attack(epsilon=1e-2,sample=None,targets=None,thismodel=None,thiscriterion=None,reduced=True, dev="cpu", filtered_indices=[i for i in range(n_input_features)],restrict_impact=-1):
    if epsilon == 0:
        return sample
    n_Vars = len(filtered_indices)
    
    wanted_full_indices = get_wanted_full_indices(filtered_indices)
    scalers = all_scalers[wanted_full_indices]
    defaults_per_variable = all_defaults[wanted_full_indices]
    scaled_defaults_per_variable = all_defaults_scaled[wanted_full_indices]
    
    device = torch.device(dev)
    
    xadv = sample.clone().detach()
    
    # inputs need to be included when calculating gradients
    xadv.requires_grad = True
    
    # from the undisturbed predictions, both the model and the criterion are already available and can be used here again;
    # it's just that they were each part of a function, so not automatically in the global scope
    if thismodel==None and thiscriterion==None:
        global model
        global criterion
    
    # forward
    preds = thismodel(xadv)
    
    loss = thiscriterion(preds, targets).mean()
    
    thismodel.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        # get sign of gradient
        dx = torch.sign(xadv.grad.detach())
        
        # add to sample
        xadv += epsilon*dx
        
        # remove the impact on selected variables (exclude integers, default values)
        # and limit perturbation based on original value
        if reduced:
            for i in range(n_Vars):
                if wanted_full_indices[i] in integer_indices:
                    xadv[:,i] = sample[:,i]
                    #print('integer index:', wanted_full_indices[i])
                else: # non integer, but might have defaults that should be excluded from shift
                    defaults = sample[:,i].cpu() == scaled_defaults_per_variable[i]
                    if torch.sum(defaults) != 0:
                        xadv[:,i][defaults] = sample[:,i][defaults]

                    if restrict_impact > 0:
                        original_back = scalers[i].inverse_transform(sample[:,i])
                        difference_back = scalers[i].inverse_transform(xadv.detach()[:,i]) - original_back
                        allowed_perturbation = restrict_impact * np.abs(original_back)
                        high_impact = np.abs(difference_back) > allowed_perturbation
                        if np.sum(high_impact)!=0:
                            scaled_back_max_perturbed = torch.from_numpy(original_back) + torch.from_numpy(allowed_perturbation) * dx[:,i]
                            xadv[high_impact,i] = torch.Tensor(scalers[i].transform(scaled_back_max_perturbed[high_impact].reshape(-1,1)).flatten())
                    
        return xadv.detach()
    
    
def syst_var(epsilon=1e-2,sample=None,reduced=True, dev="cpu", filtered_indices=[i for i in range(n_input_features)],restrict_impact=-1, up=True):
    if epsilon == 0:
        return sample
    n_Vars = len(filtered_indices)
    
    wanted_full_indices = get_wanted_full_indices(filtered_indices)
    
    scalers = all_scalers[wanted_full_indices]
    
    defaults_per_variable = all_defaults[wanted_full_indices]
    scaled_defaults_per_variable = all_defaults_scaled[wanted_full_indices]
    
    device = torch.device(dev)

    with torch.no_grad():
        # variation in common direction, default is upwards
        systvar = epsilon * torch.Tensor(np.ones((len(sample),n_Vars))).to(device)
        if up == False:
            systvar *= -1.
        # scale by a factor for individual feature
        for i in range(n_Vars):
            systvar[:,i] *= all_factor_epsilons[wanted_full_indices[i]]
        xadv = sample + systvar
        
        # use full indices and check if in int.vars. or defaults
        for i in range(n_Vars):
            if wanted_full_indices[i] in integer_indices:
                xadv[:,i] = sample[:,i]
            else: # non integer, but might have defaults that should be excluded from shift
                defaults = sample[:,i].cpu() == scaled_defaults_per_variable[i]
                if torch.sum(defaults) != 0:
                    xadv[:,i][defaults] = sample[:,i][defaults]
                    
                if restrict_impact > 0:
                    original_back = scalers[i].inverse_transform(sample[:,i])
                    difference_back = scalers[i].inverse_transform(xadv[:,i]) - original_back
                    allowed_perturbation = restrict_impact * np.abs(original_back)
                    high_impact = np.abs(difference_back) > allowed_perturbation
                    if np.sum(high_impact)!=0:
                        scaled_back_max_perturbed = torch.from_numpy(original_back[high_impact]) + torch.from_numpy(allowed_perturbation[high_impact]) * torch.sign(systvar[high_impact,i])
                        xadv[high_impact,i] = torch.Tensor(scalers[i].transform(scaled_back_max_perturbed.reshape(-1,1)).flatten())

        return xadv