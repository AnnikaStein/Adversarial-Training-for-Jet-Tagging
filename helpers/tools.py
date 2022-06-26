# get relevant common files and paths for all steps of training and evaluation

import numpy as np
import torch

user = 'um106329'

akArrays_path = '/hpcwork/' + user + '/jet_flavor_MLPhysics/dataset/akArrays/'

np_arrays_path = '/hpcwork/' + user + '/jet_flavor_MLPhysics/dataset/npArrays/'

defaults_path = '/hpcwork/' + user + '/jet_flavor_MLPhysics/dataset/defaults/'

weights_path = '/hpcwork/' + user + '/jet_flavor_MLPhysics/dataset/weights/'

preprocessed_path = '/hpcwork/' + user + '/jet_flavor_MLPhysics/dataset/preprocessed/'

eval_path = '/home/' + user + '/aisafety/jet_flavor_MLPhysics/evaluate/'

def get_splits():
    splits = []
    total = 11491971
    for k in range(0,total,50000):
        splits.append(k)
    splits.append(total)
    return splits

FALLBACK_NUM_DATASETS = len(get_splits())-1

def get_default_from_full_index(index,scaled=False):
    if scaled:
        return np.load(defaults_path+'all_scaled_defaults.npy')[index]
    else:
        return np.load(defaults_path+'all_defaults.npy')[index]

def get_all_defaults(scaled=False, old=False):
    if scaled:
        if not old: 
            return np.load(defaults_path+'all_scaled_defaults.npy')
        else: 
            return np.load(defaults_path+'all_scaled_defaults_OLD.npy')
    else:
        if not old:
            return np.load(defaults_path+'all_defaults.npy')
        else:
            return np.load(defaults_path+'all_defaults_OLD.npy')

def get_all_scalers():
    return torch.load(preprocessed_path+'all_scalers.pt')

def build_wm_parameters(wmets):
    gamma = [(((weighting_method.split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0]).split('_restrictedBy')[0] for weighting_method in wmets]
    alphaparse = [((weighting_method.split('_alpha')[-1]).split('_adv_tr_eps')[0]).split('_restrictedBy')[0] for weighting_method in wmets]
    epsilon = [(weighting_method.split('_adv_tr_eps')[-1]).split('_restrictedBy')[0] for weighting_method in wmets]
    restrictedModel = [weighting_method.split('_restrictedBy')[-1] for weighting_method in wmets]
    
    return gamma, alphaparse, epsilon, restrictedModel

def build_wm_text_dict(gamma,alphaparse,epsilon):
    wm_def_text = {'_noweighting': 'Nominal training', 
               '_ptetaflavloss' : 'Nominal training',
               '_altptetaflavloss' : 'Nominal training',
               '_ptetaflavloss_focalloss' : 'Nominal training', 
               '_altptetaflavlossfocalloss' : 'Nominal training', 
              }
    more_text = [(f'_ptetaflavloss_focalloss_gamma{g}' , 'Nominal training') for g, a in zip(gamma,alphaparse)] + \
                [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}' , 'Nominal training') for g, a in zip(gamma,alphaparse)] + \
                [(f'_ptetaflavloss_focalloss_alpha{a}' , 'Nominal training') for a in alphaparse] + \
                [(f'_altptetaflavlossfocalloss_gamma{g}' ,'Nominal training') for g in gamma] + \
                [(f'_altptetaflavlossfocalloss' , 'Nominal training') for g in gamma] + \
                [(f'_ptetaflavloss_focalloss' , 'Nominal training') for g in gamma] + \
                [(f'_altptetaflavlossfocalloss_gamma{g}_alpha{a}' , 'Nominal training') for g, a in zip(gamma,alphaparse)] + \
                [(f'_ptetaflavloss_focalloss_gamma{g}_adv_tr_eps{e}' , 'Adversarial training') for g, a, e in zip(gamma,alphaparse,epsilon)] + \
                [(f'_ptetaflavloss_focalloss_gamma{g}_adv_tr_eps{e}_restricted' , 'Adversarial training') for g, a, e in zip(gamma,alphaparse,epsilon)]

    more_text_dict = {k:v for k, v in more_text}
    wm_def_text = {**wm_def_text, **more_text_dict}
    return wm_def_text

def build_wm_color_dict(gamma,alphaparse,epsilon):
    wm_def_color = {'_noweighting': '#92638C', 
               '_ptetaflavloss' : '#F06644',
               '_altptetaflavloss' : '#7AC7A3',
               '_ptetaflavloss_focalloss' : '#FEC55C', 
               '_altptetaflavlossfocalloss' : '#4BC2D8',
              }
    more_color = [(f'_ptetaflavloss_focalloss_gamma{g}' , '#FEC55C') for g in gamma] + \
                 [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}' , '#FEC55C') for g, a in zip(gamma,alphaparse)] + \
                 [(f'_ptetaflavloss_focalloss_alpha{a}' , '#FEC55C') for a in alphaparse] + \
                 [(f'_altptetaflavlossfocalloss_gamma{g}' , '#4BC2D8') for g in gamma] + \
                 [(f'_altptetaflavlossfocalloss_gamma{g}_alpha{a}' , '#4BC2D8') for g, a in zip(gamma,alphaparse)] + \
                 [(f'_ptetaflavloss_focalloss_gamma{g}_adv_tr_eps{e}' , '#FEC55C') for g, e in zip(gamma,epsilon)] + \
                 [(f'_ptetaflavloss_focalloss_gamma{g}_alpha{a}_adv_tr_eps{e}' , '#FEC55C') for g, a, e in zip(gamma,alphaparse,epsilon)] + \
                 [(f'_ptetaflavloss_adv_tr_eps{e}' , '#FEC55C') for e in epsilon] + \
                 [(f'_altptetaflavlossfocalloss' , '#4BC2D8') for g in gamma] + \
                 [(f'_ptetaflavloss_focalloss' , '#FEC55C') for g in gamma]

    more_color_dict = {k:v for k, v in more_color}
    wm_def_color =  {**wm_def_color, **more_color_dict}
    return wm_def_color