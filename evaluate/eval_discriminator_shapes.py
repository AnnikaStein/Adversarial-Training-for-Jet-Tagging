import numpy as np
np.seterr(all="ignore")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgba, to_rgb, hex2color
import mplhep as hep
from cycler import cycler
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

from sklearn import metrics

import gc

import coffea.hist as hist

import time

import argparse

import sys

from scipy.stats import ks_2samp
from scipy.stats import entropy

sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/attack/")
from disturb_inputs import fgsm_attack, apply_noise
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/training/")
from focal_loss import FocalLoss, focal_loss
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/helpers/")
from tools import preprocessed_path, eval_path, FALLBACK_NUM_DATASETS, build_wm_text_dict, build_wm_color_dict, build_wm_parameters
from variables import input_indices_wanted

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])

plt.ioff()


parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", help="Number of previously trained epochs, can be a comma-separated list")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _altptetaflavloss or with additional _focalloss")
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("check_inputs", help="Check certain inputs in slices of Prob(udsg) (yes/no)") # currently not used
parser.add_argument("disturb_setup", help="Can be raw (no perturbation at all), epsX or sigmaX where X is some positive value for FGSM or Noise, respectively.")
parser.add_argument("restrict", help="Restrict impact of the attack ? -1 for no, some positive value for yes")
args = parser.parse_args()



NUM_DATASETS = args.files
NUM_DATASETS = FALLBACK_NUM_DATASETS if NUM_DATASETS < 0 else NUM_DATASETS
at_epoch = args.prevep
epochs = [int(e) for e in at_epoch.split(',')]
weighting_method = args.wm
wmets = [w for w in weighting_method.split(',_')]
n_samples = args.jets
compare_eps = True if len(epochs) > 1 else False
compare_wmets = True if len(wmets) > 1 else False
check_inputs = args.check_inputs

print(f'Evaluate training at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')

gammaModel, alphaparseModel, epsilonModel, restrictedModel = build_wm_parameters(wmets)
print('gammaModel',gammaModel)
print('alphaModel',alphaparseModel)
print('epsilonModel',epsilonModel)

wm_def_text = build_wm_text_dict(gammaModel,alphaparseModel,epsilonModel)

wm_def_color =  build_wm_color_dict(gammaModel,alphaparseModel,epsilonModel)

colorcode = ['#B22222','#00FFFF','#006400']
colorcode_2 = ['#DA7479','#63D8F1','#7DFDB4']  # from http://tristen.ca/hcl-picker/#/hlc/4/1/DA7479/7DFDB4


disturb_setup = args.disturb_setup
sig, eps = 0, 0
if disturb_setup != 'raw':
    if 'sigma' in disturb_setup:
        sig = float(disturb_setup.split('sigma')[-1])
    elif 'eps' in disturb_setup:
        eps = float(disturb_setup.split('eps')[-1])
        # when comparing before / after attack, one needs the loss function for FGSM
        if ('_ptetaflavloss' in wmets[0]) or ('_altptetaflavloss' in wmets[0]):
            if 'focalloss' not in wmets[0]:
                criterion = nn.CrossEntropyLoss(reduction='none')
            elif 'focalloss' in wmets[0]:
                if 'alpha' not in wmets[0]:
                    alpha = None
                else:
                    commasep_alpha = [a for a in (alphaparseModel[0]).split(',')]
                    alpha = torch.Tensor([float(commasep_alpha[0]),float(commasep_alpha[1]),float(commasep_alpha[2])]).to(device)
                if 'gamma' not in wmets[0]:
                    gamma = 2.0
                else:
                    gamma = float(gammaModel[0])
                criterion = FocalLoss(alpha, gamma, reduction='none')
        else:
            criterion = nn.CrossEntropyLoss()
            
    restrict = float(args.restrict)
    restrict_text = f'_restrictedBy{restrict}' if restrict > 0 else '_restrictedByInf'
    print('perturbation',restrict_text)
else:
    restrict_text = ''
        
    


# 51 bin edges betweeen 0 and 1 --> 50 bins of width 0.02, plus two additional bins at -0.05 and -0.025, as well as at 1.025 and 1.05
# in total: 54 bins, 55 bin edges
#bins = np.append(np.insert(np.linspace(0,0.98,50),0,[-0.05,-0.025]),[1.00001,1.025,1.05])
bins = np.linspace(0.,1.,50)
#print(bins)
#sys.exit()
# Loading data will be necessary for all use cases

test_input_file_paths = [preprocessed_path + f'test_inputs_%d.pt' % k for k in range(NUM_DATASETS)]
test_target_file_paths = [preprocessed_path + f'test_targets_%d.pt' % k for k in range(NUM_DATASETS)]

# note: the default case without parameters in the function input_indices_wanted return all high-level variables, as well as 28 (all) features for the first 6 tracks
used_variables = input_indices_wanted()
slices = torch.LongTensor(used_variables)
# use n_input_features as the number of inputs to the model (later)
n_input_features = len(slices)

'''

    Load inputs and targets
    
'''

test_inputs = torch.cat(tuple(torch.load(ti)[:,slices] for ti in test_input_file_paths)).float()
print('test inputs done')
len_test = len(test_inputs)
print('number of test inputs', len_test)


test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths))
print('test targets done')

jetFlavour = test_targets+1


def calc_BvL(predictions):
    matching_targets = test_targets
    matching_predictions = predictions
    
    custom_BvL = np.where(((matching_predictions[:,0]+matching_predictions[:,2]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1), (matching_predictions[:,0])/(matching_predictions[:,0]+matching_predictions[:,2]), (-0.045)*np.ones(len(matching_targets)))
    
    return custom_BvL

def calc_BvC(predictions):
    matching_targets = test_targets
    matching_predictions = predictions
    
    custom_BvC = np.where(((matching_predictions[:,0]+matching_predictions[:,1]) != 0) & (matching_predictions[:,0] >= 0) & (matching_predictions[:,0] <= 1), (matching_predictions[:,0])/(matching_predictions[:,0]+matching_predictions[:,1]), (-0.045)*np.ones(len(matching_targets)))
  
    return custom_BvC
    
def calc_CvB(predictions):
    matching_targets = test_targets
    matching_predictions = predictions
    
    custom_CvB = np.where(((matching_predictions[:,0]+matching_predictions[:,1]) != 0) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1), (matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]), (-0.045)*np.ones(len(matching_targets)))
        
    return custom_CvB
    
def calc_CvL(predictions):
    matching_targets = test_targets
    matching_predictions = predictions
    
    custom_CvL = np.where(((matching_predictions[:,0]+matching_predictions[:,2]) != 0) & (matching_predictions[:,1] >= 0) & (matching_predictions[:,1] <= 1), (matching_predictions[:,1])/(matching_predictions[:,1]+matching_predictions[:,2]), (-0.045)*np.ones(len(matching_targets)))
    
    return custom_CvL







# =============================================================================================================================
#
#
#                                            compare epochs or weighting methods
#
#
# -----------------------------------------------------------------------------------------------------------------------------

# create model, similar for all epochs / weighting methods and only load the weights during the loop
# everything is in evaluation mode, and no gradients are necessary in this script

if compare_eps or compare_wmets:
    pass
    # ToDo
else:
    
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

    # this is just a very quick plot of one epoch, one weighting method only

    checkpoint = torch.load(f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{weighting_method}_{NUM_DATASETS}_{n_samples}/model_{at_epoch}_epochs{weighting_method}_{NUM_DATASETS}_datasets_{n_samples}.pt', map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions = model(test_inputs).detach().numpy()
    if sig != 0:
        disturbed_predictions = model(apply_noise(test_inputs,sig,filtered_indices=used_variables,restrict_impact=restrict)).detach().numpy()
        wm_text_extra = f'Noise $\sigma={sig}$'
    elif eps != 0:
        disturbed_predictions = model(fgsm_attack(eps,test_inputs,test_targets,model,criterion,dev=device,filtered_indices=used_variables,restrict_impact=restrict)).detach().numpy()
        wm_text_extra = f'FGSM $\epsilon={eps}$'
        
    wm_text = wm_def_text[weighting_method]


    mostprob = np.argmax(predictions, axis=-1)
    cfm = metrics.confusion_matrix(test_targets.cpu(), mostprob)
    print(f'epoch {at_epoch}\n',cfm)
    with open(eval_path + f'confusion_matrices/{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.npy', 'wb') as f:
        np.save(f, cfm)

    minimum = 0.
    maximum = 1.
    edges = np.linspace(0,1.,51)
    ratio_bins = np.linspace(minimum+(maximum-minimum)/50/2,maximum-(maximum-minimum)/50/2,50)
    #print(edges)
    #print(ratio_bins)
    classifierHist = hist.Hist("Jets / 0.02 units",
                        hist.Cat("sample","sample name"),
                        hist.Cat("flavour","flavour of the jet"),
                        hist.Bin("probb","P(b)",edges),
                        hist.Bin("probc","P(c)",edges),
                        hist.Bin("probudsg","P(udsg)",edges),
                     )

    classifierHist.fill(sample=wm_text,flavour='b-jets',probb=predictions[:,0][jetFlavour==1],probc=predictions[:,1][jetFlavour==1],probudsg=predictions[:,2][jetFlavour==1])
    classifierHist.fill(sample=wm_text,flavour='c-jets',probb=predictions[:,0][jetFlavour==2],probc=predictions[:,1][jetFlavour==2],probudsg=predictions[:,2][jetFlavour==2])
    classifierHist.fill(sample=wm_text,flavour='udsg-jets',probb=predictions[:,0][jetFlavour==3],probc=predictions[:,1][jetFlavour==3],probudsg=predictions[:,2][jetFlavour==3])

    opacity_raw = 1.
    if disturb_setup != 'raw':
        opacity_raw = 0.42
        classifierHist.fill(sample=wm_text_extra,flavour='b-jets',probb=disturbed_predictions[:,0][jetFlavour==1],probc=disturbed_predictions[:,1][jetFlavour==1],probudsg=disturbed_predictions[:,2][jetFlavour==1])
        classifierHist.fill(sample=wm_text_extra,flavour='c-jets',probb=disturbed_predictions[:,0][jetFlavour==2],probc=disturbed_predictions[:,1][jetFlavour==2],probudsg=disturbed_predictions[:,2][jetFlavour==2])
        classifierHist.fill(sample=wm_text_extra,flavour='udsg-jets',probb=disturbed_predictions[:,0][jetFlavour==3],probc=disturbed_predictions[:,1][jetFlavour==3],probudsg=disturbed_predictions[:,2][jetFlavour==3])
        
        fig1, (ax1,ax1R) = plt.subplots(2,1,sharex=True,figsize=[6,6],gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0})
        fig2, (ax2,ax2R) = plt.subplots(2,1,sharex=True,figsize=[6,6],gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0})
        fig3, (ax3,ax3R) = plt.subplots(2,1,sharex=True,figsize=[6,6],gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0})
        
        ax1R.set_prop_cycle(cycler(color=colorcode))
        ax2R.set_prop_cycle(cycler(color=colorcode))
        ax3R.set_prop_cycle(cycler(color=colorcode))

    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[24,6])
        plt.subplots_adjust(wspace=0.3)
    
    ax1.set_prop_cycle(cycler(color=colorcode))
    ax2.set_prop_cycle(cycler(color=colorcode))
    ax3.set_prop_cycle(cycler(color=colorcode))
        
    # =================================================================================================================
    # 
    #                                           Plot outputs (not stacked)
    # 
    # -----------------------------------------------------------------------------------------------------------------
    # split per flavour outputs

    
    if disturb_setup != 'raw':
        custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('sample','probc','probudsg'),overlay='flavour',ax=ax1,clear=False,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probudsg'),overlay='flavour',ax=ax2,clear=False,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probc'),overlay='flavour',ax=ax3,clear=False,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
    else:
        custom_ax1 = hist.plot1d(classifierHist[wm_text].sum('sample','probc','probudsg'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linewidth':2,'alpha':1})
        custom_ax2 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probudsg'),overlay='flavour',ax=ax2,clear=False,line_opts={'color':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linewidth':2,'alpha':1})
        custom_ax3 = hist.plot1d(classifierHist[wm_text].sum('sample','probb','probc'),overlay='flavour',ax=ax3,clear=False,line_opts={'color':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linewidth':2,'alpha':1})
    # MLPhysics
    #ax1.set_prop_cycle(None)
    #ax2.set_prop_cycle(None)
    #ax3.set_prop_cycle(None)
    if disturb_setup != 'raw':
        hist.plot1d(classifierHist[wm_text_extra].sum('sample','probc','probudsg'),overlay='flavour',ax=custom_ax1,clear=False,line_opts={'color':colorcode,'linewidth':2,'alpha':1})
        hist.plot1d(classifierHist[wm_text_extra].sum('sample','probb','probudsg'),overlay='flavour',ax=custom_ax2,clear=False,line_opts={'color':colorcode,'linewidth':2,'alpha':1})
        hist.plot1d(classifierHist[wm_text_extra].sum('sample','probb','probc'),overlay='flavour',ax=custom_ax3,clear=False,line_opts={'color':colorcode,'linewidth':2,'alpha':1})
        if sig > 0:
            ax3.legend([f'b ($\sigma$ = {sig})',f'c ($\sigma$ = {sig})',f'udsg ($\sigma$ = {sig})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=41,loc='upper right')
            ratio_label = 'Noise/raw'
        if eps > 0:
            ax3.legend([f'b ($\epsilon$ = {eps})',f'c ($\epsilon$ = {eps})',f'udsg ($\epsilon$ = {eps})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=1,loc='upper right')
            ratio_label = 'FGSM/raw'
            
        for p in range(3):
            if p == 0:
                numB  = classifierHist[wm_text_extra].sum('sample','probc','probudsg')['b-jets'].sum('flavour').values()[()]
                numC  = classifierHist[wm_text_extra].sum('sample','probc','probudsg')['c-jets'].sum('flavour').values()[()]
                numL  = classifierHist[wm_text_extra].sum('sample','probc','probudsg')['udsg-jets'].sum('flavour').values()[()]
                denomB  = classifierHist[wm_text].sum('sample','probc','probudsg')['b-jets'].sum('flavour').values()[()]
                denomC  = classifierHist[wm_text].sum('sample','probc','probudsg')['c-jets'].sum('flavour').values()[()]
                denomL  = classifierHist[wm_text].sum('sample','probc','probudsg')['udsg-jets'].sum('flavour').values()[()]
                
            elif p == 1:
                numB  = classifierHist[wm_text_extra].sum('sample','probb','probudsg')['b-jets'].sum('flavour').values()[()]
                numC  = classifierHist[wm_text_extra].sum('sample','probb','probudsg')['c-jets'].sum('flavour').values()[()]
                numL  = classifierHist[wm_text_extra].sum('sample','probb','probudsg')['udsg-jets'].sum('flavour').values()[()]
                denomB  = classifierHist[wm_text].sum('sample','probb','probudsg')['b-jets'].sum('flavour').values()[()]
                denomC  = classifierHist[wm_text].sum('sample','probb','probudsg')['c-jets'].sum('flavour').values()[()]
                denomL  = classifierHist[wm_text].sum('sample','probb','probudsg')['udsg-jets'].sum('flavour').values()[()]
                
            elif p == 2:
                numB  = classifierHist[wm_text_extra].sum('sample','probb','probc')['b-jets'].sum('flavour').values()[()]
                numC  = classifierHist[wm_text_extra].sum('sample','probb','probc')['c-jets'].sum('flavour').values()[()]
                numL  = classifierHist[wm_text_extra].sum('sample','probb','probc')['udsg-jets'].sum('flavour').values()[()]
                denomB  = classifierHist[wm_text].sum('sample','probb','probc')['b-jets'].sum('flavour').values()[()]
                denomC  = classifierHist[wm_text].sum('sample','probb','probc')['c-jets'].sum('flavour').values()[()]
                denomL  = classifierHist[wm_text].sum('sample','probb','probc')['udsg-jets'].sum('flavour').values()[()]
                
                
            num = numB + numC + numL
            denom = denomB + denomC + denomL
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


            #print(len(bins), len(ratio_bins), len(edges), len(ratioB), len(ratio_errB))
            #sys.exit()

            #print(ratioB)
            #print(ratio_errB)

            for f, flav in enumerate(['B','C','L']):
                exec(f"ax{p+1}R.errorbar(ratio_bins,ratio{flav},yerr=ratio_err{flav},fmt='.',color=colorcode[f])")
            exec(f"ax{p+1}R.set_ylabel(ratio_label,fontsize=21)")
              
    else:
        ax3.legend(loc='upper right',ncol=1,fontsize=18)
        
    
    ax1.get_legend().remove(), ax2.get_legend().remove()

    ax1.set_ylim(bottom=0, auto=True)
    ax2.set_ylim(bottom=0, auto=True)
    ax3.set_ylim(bottom=0, auto=True)
    
    if disturb_setup != 'raw':
        ax1R.set_ylim(0,2)
        ax2R.set_ylim(0,2)
        ax3R.set_ylim(0,2)
        ax1R.plot([minimum,maximum],[1,1],color='black')    
        ax1R.set_xlim(minimum,maximum)
        ax2R.plot([minimum,maximum],[1,1],color='black')    
        ax2R.set_xlim(minimum,maximum)
        ax3R.plot([minimum,maximum],[1,1],color='black')    
        ax3R.set_xlim(minimum,maximum)
    
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    ax1.autoscale(True)
    ax2.autoscale(True)
    ax3.autoscale(True)
    
    if disturb_setup != 'raw':
        xlbl = ax1.get_xlabel()
        ax1R.set_xlabel(xlbl) 
        xlbl = ax2.get_xlabel()
        ax2R.set_xlabel(xlbl) 
        xlbl = ax3.get_xlabel()
        ax3R.set_xlabel(xlbl) 
    
    # new, MLPhysics
    # https://stackoverflow.com/a/8482667
    # old with epoch label
    #ax2.text(0.9,0.85,f'{wm_text},\nepoch {at_epoch}',fontsize=15, ha='right', transform=ax2.transAxes)
    # new without epoch label (paper)
    #ax2.text(0.9,0.85,f'{wm_text}',fontsize=15, ha='right', transform=ax2.transAxes)
    # placed a bit higher
    ax2.text(0.9,0.85,f'{wm_text}',fontsize=15, ha='right', transform=ax2.transAxes)
    #fig.suptitle(f'Classifier outputs, {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets')
    
    if disturb_setup == 'raw':
        fig.savefig(eval_path + f'discriminator_shapes/shapes_new/{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=400)
        fig.savefig(eval_path + f'discriminator_shapes/shapes_new/{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig.savefig(eval_path + f'discriminator_shapes/shapes_new/{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
    else:    
        fig1.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbB{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=600)
        fig1.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbB{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig1.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbB{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
        fig2.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbC{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=600)
        fig2.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbC{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig2.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbC{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
        fig3.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbL{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=600)
        fig3.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbL{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig3.savefig(eval_path + f'discriminator_shapes/shapes_new/ProbL{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
    
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)

    # debug
    #sys.exit()

    del classifierHist
    gc.collect()



    # =================================================================================================================
    # 
    #                                     Plot discriminator shapes (not stacked)
    # 
    # -----------------------------------------------------------------------------------------------------------------


    custom_BvL = calc_BvL(predictions)
    custom_BvC = calc_BvC(predictions)
    custom_CvB = calc_CvB(predictions)
    custom_CvL = calc_CvL(predictions)

    if check_inputs != 'yes':
        del predictions
        gc.collect()
        
    if disturb_setup != 'raw':
        disturbed_BvL = calc_BvL(disturbed_predictions)
        disturbed_BvC = calc_BvC(disturbed_predictions)
        disturbed_CvB = calc_CvB(disturbed_predictions)
        disturbed_CvL = calc_CvL(disturbed_predictions)

        if check_inputs != 'yes':
            del disturbed_predictions
            gc.collect()
    
   
    discriminatorHist = hist.Hist("Jets / 0.02 units",
                        hist.Cat("sample","sample name"),
                        hist.Cat("flavour","flavour of the jet"),
                        hist.Bin("bvl","B vs L",edges),
                        hist.Bin("bvc","B vs C",edges),
                        hist.Bin("cvb","C vs B",edges),
                        hist.Bin("cvl","C vs L",edges),
                     )

    discriminatorHist.fill(sample=wm_text,flavour='b-jets',bvl=custom_BvL[jetFlavour==1],bvc=custom_BvC[jetFlavour==1],cvb=custom_CvB[jetFlavour==1],cvl=custom_CvL[jetFlavour==1])
    discriminatorHist.fill(sample=wm_text,flavour='c-jets',bvl=custom_BvL[jetFlavour==2],bvc=custom_BvC[jetFlavour==2],cvb=custom_CvB[jetFlavour==2],cvl=custom_CvL[jetFlavour==2])
    discriminatorHist.fill(sample=wm_text,flavour='udsg-jets',bvl=custom_BvL[jetFlavour==3],bvc=custom_BvC[jetFlavour==3],cvb=custom_CvB[jetFlavour==3],cvl=custom_CvL[jetFlavour==3])

    if disturb_setup != 'raw':
        discriminatorHist.fill(sample=wm_text_extra,flavour='b-jets',bvl=disturbed_BvL[jetFlavour==1],bvc=disturbed_BvC[jetFlavour==1],cvb=disturbed_CvB[jetFlavour==1],cvl=disturbed_CvL[jetFlavour==1])
        discriminatorHist.fill(sample=wm_text_extra,flavour='c-jets',bvl=disturbed_BvL[jetFlavour==2],bvc=disturbed_BvC[jetFlavour==2],cvb=disturbed_CvB[jetFlavour==2],cvl=disturbed_CvL[jetFlavour==2])
        discriminatorHist.fill(sample=wm_text_extra,flavour='udsg-jets',bvl=disturbed_BvL[jetFlavour==3],bvc=disturbed_BvC[jetFlavour==3],cvb=disturbed_CvB[jetFlavour==3],cvl=disturbed_CvL[jetFlavour==3])

        #fig, ((ax1, ax2), (ax1R, ax2R), (ax3, ax4), (ax3R, ax4R)) = plt.subplots(4, 2, figsize=[17,15],num=30)
        
        #fig = plt.figure(figsize=[17,16])
        fig1, (ax1,ax1R) = plt.subplots(2,1,sharex=True,figsize=[6,6],gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0})
        fig2, (ax2,ax2R) = plt.subplots(2,1,sharex=True,figsize=[6,6],gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0})
        fig3, (ax3,ax3R) = plt.subplots(2,1,sharex=True,figsize=[6,6],gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0})
        fig4, (ax4,ax4R) = plt.subplots(2,1,sharex=True,figsize=[6,6],gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0})
        
        ax1R.set_prop_cycle(cycler(color=colorcode))
        ax2R.set_prop_cycle(cycler(color=colorcode))
        ax3R.set_prop_cycle(cycler(color=colorcode))
        ax4R.set_prop_cycle(cycler(color=colorcode))
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[17,15],num=30)
        
        
    ax1.set_prop_cycle(cycler(color=colorcode))
    ax2.set_prop_cycle(cycler(color=colorcode))
    ax3.set_prop_cycle(cycler(color=colorcode))
    ax4.set_prop_cycle(cycler(color=colorcode))
    #plt.subplots_adjust(wspace=0.4)
    #custom_ax1 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl'),overlay='flavour',ax=ax1,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linestyle':':','edgecolor':[to_rgba(colorcode[i],1) for i in range(3)],'linewidth':2})
    #custom_ax2 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl'),overlay='flavour',ax=ax2,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linestyle':':','edgecolor':[to_rgba(colorcode[i],1) for i in range(3)],'linewidth':2})
    #custom_ax3 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl'),overlay='flavour',ax=ax3,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linestyle':':','edgecolor':[to_rgba(colorcode[i],1) for i in range(3)],'linewidth':2})
    #custom_ax4 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb'),overlay='flavour',ax=ax4,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linestyle':':','edgecolor':[to_rgba(colorcode[i],1) for i in range(3)],'linewidth':2})
    
    if disturb_setup != 'raw':
        custom_ax1 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl'),overlay='flavour',ax=ax1,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        custom_ax2 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl'),overlay='flavour',ax=ax2,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        custom_ax3 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl'),overlay='flavour',ax=ax3,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
        custom_ax4 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb'),overlay='flavour',ax=ax4,clear=False,line_opts=None,fill_opts={'facecolor':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)]})
    else:
        custom_ax1 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl'),overlay='flavour',ax=ax1,clear=False,fill_opts=None,line_opts={'color':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linewidth':2,'alpha':1})
        custom_ax2 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl'),overlay='flavour',ax=ax2,clear=False,fill_opts=None,line_opts={'color':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linewidth':2,'alpha':1})
        custom_ax3 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl'),overlay='flavour',ax=ax3,clear=False,fill_opts=None,line_opts={'color':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linewidth':2,'alpha':1})
        custom_ax4 = hist.plot1d(discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb'),overlay='flavour',ax=ax4,clear=False,fill_opts=None,line_opts={'color':[to_rgba(colorcode[i],opacity_raw/2) for i in range(3)],'linewidth':2,'alpha':1})
    # gamma25
    #ax1.legend(loc=(0.67,0.7),ncol=1,fontsize=13.5)
    #ax3.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()
    # gamma2
    #ax1.legend(loc='upper center',ncol=1,fontsize=13.5)
    #ax3.get_legend().remove(), ax2.get_legend().remove(), ax4.get_legend().remove()
    # new MLPhysics
    # https://stackoverflow.com/a/20049202
    if disturb_setup != 'raw':
        hist.plot1d(discriminatorHist[wm_text_extra].sum('sample','bvc','cvb','cvl'),overlay='flavour',ax=ax1,clear=False,line_opts={'color':colorcode,'linewidth':2,'alpha':1})
        hist.plot1d(discriminatorHist[wm_text_extra].sum('sample','bvl','cvb','cvl'),overlay='flavour',ax=ax2,clear=False,line_opts={'color':colorcode,'linewidth':2,'alpha':1})
        hist.plot1d(discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvl'),overlay='flavour',ax=ax3,clear=False,line_opts={'color':colorcode,'linewidth':2,'alpha':1})
        hist.plot1d(discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvb'),overlay='flavour',ax=ax4,clear=False,line_opts={'color':colorcode,'linewidth':2,'alpha':1})
        if sig > 0:
            ax4.legend([f'b ($\sigma$ = {sig})',f'c ($\sigma$ = {sig})',f'udsg ($\sigma$ = {sig})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=1,loc='upper left', borderpad=2)
            ratio_label = 'Noise/raw'
        if eps > 0:
            ax4.legend([f'b ($\epsilon$ = {eps})',f'c ($\epsilon$ = {eps})',f'udsg ($\epsilon$ = {eps})','b (raw)','c (raw)','udsg (raw)'],fontsize=12,ncol=1,loc='upper left', borderpad=2)
            ratio_label = 'FGSM/raw'
        
        for p in range(4):
            if p == 0:
                numB  = discriminatorHist[wm_text_extra].sum('sample','bvc','cvb','cvl')['b-jets'].sum('flavour').values()[()]
                numC  = discriminatorHist[wm_text_extra].sum('sample','bvc','cvb','cvl')['c-jets'].sum('flavour').values()[()]
                numL  = discriminatorHist[wm_text_extra].sum('sample','bvc','cvb','cvl')['udsg-jets'].sum('flavour').values()[()]
                denomB  = discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl')['b-jets'].sum('flavour').values()[()]
                denomC  = discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl')['c-jets'].sum('flavour').values()[()]
                denomL  = discriminatorHist[wm_text].sum('sample','bvc','cvb','cvl')['udsg-jets'].sum('flavour').values()[()]
                
            elif p == 1:
                numB  = discriminatorHist[wm_text_extra].sum('sample','bvl','cvb','cvl')['b-jets'].sum('flavour').values()[()]
                numC  = discriminatorHist[wm_text_extra].sum('sample','bvl','cvb','cvl')['c-jets'].sum('flavour').values()[()]
                numL  = discriminatorHist[wm_text_extra].sum('sample','bvl','cvb','cvl')['udsg-jets'].sum('flavour').values()[()]
                denomB  = discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl')['b-jets'].sum('flavour').values()[()]
                denomC  = discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl')['c-jets'].sum('flavour').values()[()]
                denomL  = discriminatorHist[wm_text].sum('sample','bvl','cvb','cvl')['udsg-jets'].sum('flavour').values()[()]
                
            elif p == 2:
                numB  = discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvl')['b-jets'].sum('flavour').values()[()]
                numC  = discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvl')['c-jets'].sum('flavour').values()[()]
                numL  = discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvl')['udsg-jets'].sum('flavour').values()[()]
                denomB  = discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl')['b-jets'].sum('flavour').values()[()]
                denomC  = discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl')['c-jets'].sum('flavour').values()[()]
                denomL  = discriminatorHist[wm_text].sum('sample','bvl','bvc','cvl')['udsg-jets'].sum('flavour').values()[()]
                
            elif p == 3:
                numB  = discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvb')['b-jets'].sum('flavour').values()[()]
                numC  = discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvb')['c-jets'].sum('flavour').values()[()]
                numL  = discriminatorHist[wm_text_extra].sum('sample','bvl','bvc','cvb')['udsg-jets'].sum('flavour').values()[()]
                denomB  = discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb')['b-jets'].sum('flavour').values()[()]
                denomC  = discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb')['c-jets'].sum('flavour').values()[()]
                denomL  = discriminatorHist[wm_text].sum('sample','bvl','bvc','cvb')['udsg-jets'].sum('flavour').values()[()]
                
            num = numB + numC + numL
            denom = denomB + denomC + denomL
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


            #print(len(bins), len(ratio_bins), len(edges), len(ratioB), len(ratio_errB))
            #sys.exit()

            #print(ratioB)
            #print(ratio_errB)

            for f, flav in enumerate(['B','C','L']):
                exec(f"ax{p+1}R.errorbar(ratio_bins,ratio{flav},yerr=ratio_err{flav},fmt='.',color=colorcode[f])")
            exec(f"ax{p+1}R.set_ylabel(ratio_label,fontsize=21)")
            
    else:
        ax4.legend(loc='upper left',ncol=1,fontsize=13.5, borderpad=2)
        
    ax1.get_legend().remove(), ax2.get_legend().remove(), ax3.get_legend().remove()

    ax1.set_ylim(bottom=0, auto=True)
    ax2.set_ylim(bottom=0, auto=True)
    ax3.set_ylim(bottom=0, auto=True)
    ax4.set_ylim(bottom=0, auto=True)

    if disturb_setup != 'raw':
        ax1R.set_ylim(0,2)
        ax2R.set_ylim(0,2)
        ax3R.set_ylim(0,2)
        ax4R.set_ylim(0,2)
        ax1R.plot([minimum,maximum],[1,1],color='black')    
        ax1R.set_xlim(minimum,maximum)
        ax2R.plot([minimum,maximum],[1,1],color='black')    
        ax2R.set_xlim(minimum,maximum)
        ax3R.plot([minimum,maximum],[1,1],color='black')    
        ax3R.set_xlim(minimum,maximum)
        ax4R.plot([minimum,maximum],[1,1],color='black')    
        ax4R.set_xlim(minimum,maximum)

        xlbl = ax1.get_xlabel()
        ax1R.set_xlabel(xlbl) 
        xlbl = ax2.get_xlabel()
        ax2R.set_xlabel(xlbl) 
        xlbl = ax3.get_xlabel()
        ax3R.set_xlabel(xlbl) 
        xlbl = ax4.get_xlabel()
        ax4R.set_xlabel(xlbl)
    
    
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')

    ax1.autoscale(True)
    ax2.autoscale(True)
    ax3.autoscale(True)
    ax4.autoscale(True)

    #ax1.ticklabel_format(scilimits=(-5,5))
    #ax2.ticklabel_format(scilimits=(-5,5))
    #ax3.ticklabel_format(scilimits=(-5,5))
    #ax4.ticklabel_format(scilimits=(-5,5))
    # for adversarial training gamma25
    #ax4.text(0.49,5e5,f'{wm_text},\nepoch {at_epoch}',fontsize=14)
    # for basic training gamma25
    #ax4.text(0.59,5e5,f'{wm_text},\nepoch {at_epoch}',fontsize=14)
    # for basic training gamma2
    #ax2.text(0.33,5e6,f'{wm_text}, epoch {at_epoch}',fontsize=14)
    # new, MLPhysics
    # old (with epoch label)
    #ax1.text(0.42,0.1,f'{wm_text},\nepoch {at_epoch}',fontsize=14, transform=ax1.transAxes)
    # new (no epoch label, paper)
    # ax1.text(0.42,0.1,f'{wm_text}',fontsize=14, transform=ax1.transAxes)
    # placed a bit higher
    ax1.text(0.52,0.85,f'{wm_text}',fontsize=14, transform=ax1.transAxes)
    #fig.suptitle(f'Discriminators, {wm_text}\nAfter {at_epoch} epochs, evaluated on {len_test} jets')
    
    if disturb_setup == 'raw':
        fig.savefig(eval_path + f'discriminator_shapes/shapes_new/discriminators_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=400)
        fig.savefig(eval_path + f'discriminator_shapes/shapes_new/discriminators_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig.savefig(eval_path + f'discriminator_shapes/shapes_new/discriminators_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
    else:
        fig1.savefig(eval_path + f'discriminator_shapes/shapes_new/BvL_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=600)
        fig1.savefig(eval_path + f'discriminator_shapes/shapes_new/BvL_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig1.savefig(eval_path + f'discriminator_shapes/shapes_new/BvL_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
        fig2.savefig(eval_path + f'discriminator_shapes/shapes_new/BvC_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=600)
        fig2.savefig(eval_path + f'discriminator_shapes/shapes_new/BvC_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig2.savefig(eval_path + f'discriminator_shapes/shapes_new/BvC_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
        fig3.savefig(eval_path + f'discriminator_shapes/shapes_new/CvB_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=600)
        fig3.savefig(eval_path + f'discriminator_shapes/shapes_new/CvB_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig3.savefig(eval_path + f'discriminator_shapes/shapes_new/CvB_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
        fig4.savefig(eval_path + f'discriminator_shapes/shapes_new/CvL_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.png', bbox_inches='tight', dpi=600)
        fig4.savefig(eval_path + f'discriminator_shapes/shapes_new/CvL_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.pdf', bbox_inches='tight')
        fig4.savefig(eval_path + f'discriminator_shapes/shapes_new/CvL_versus{weighting_method}_at_{at_epoch}_{len_test}_jetsTr_{NUM_DATASETS}_files_{n_samples}_samples_{disturb_setup}{restrict_text}_v2.svg', bbox_inches='tight')
    gc.collect()
    plt.show(block=False)
    time.sleep(5)
    plt.clf()
    plt.cla()
    plt.close('all')
    gc.collect(2)