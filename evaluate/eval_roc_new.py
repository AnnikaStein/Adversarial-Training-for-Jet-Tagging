import numpy as np

import matplotlib.pyplot as plt
import mplhep as hep

import torch
import torch.nn as nn

from sklearn import metrics

import gc

import coffea.hist as hist

import time

import argparse

import sys

sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/attack/")
from disturb_inputs import fgsm_attack, apply_noise, syst_var
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/training/")
from focal_loss import FocalLoss
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/helpers/")
from tools import preprocessed_path, eval_path, FALLBACK_NUM_DATASETS, build_wm_text_dict
from variables import input_indices_wanted

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#This is just some plot styling
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])


parser = argparse.ArgumentParser(description="Setup for evaluation")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("prevep", help="Number of previously trained epochs, can be a comma-separated list")
parser.add_argument("comparesetup", help="Setup for comparison, examples: BvL_raw, CvB_sigma0.01, c_eps0.01, BvL_sys0.01, can be a comma-separated list")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _altptetaflavloss or with additional _focalloss; specifying multiple comma-separated weighting methods is possible")
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("add_axis", help="Add a second axis as inset to the plot")
parser.add_argument("FGSM_setup", help="If FGSM, create adversarial inputs from individual models (-1) or only for a given setup (e.g. BasicAt497 (one model, fixed epoch), Adversarial (one model, indiv. epochs), At42 (fixed epoch, but for each model individually)")
parser.add_argument("log_axis", help="Flip axis and use log scale (yes/no)")
parser.add_argument("force_compare", help="Force compare (yes/no)")
parser.add_argument("line_thickness", help="Line thickness")
parser.add_argument("save_mode", help="Save AUC only, or save plots of ROC curves (ROC/AUC), only useful for multiple specified setups (compare).")
parser.add_argument("restrict", help="Restrict impact of the attack ? -1 for no, some positive value for yes")
args = parser.parse_args()

NUM_DATASETS = args.files
NUM_DATASETS = FALLBACK_NUM_DATASETS if NUM_DATASETS < 0 else NUM_DATASETS
at_epoch = args.prevep
epochs = [int(e) for e in at_epoch.split(',')]
compare_setup = args.comparesetup
setups = [s for s in compare_setup.split(',')]
weighting_method = args.wm
wmets = [w for w in weighting_method.split(',')]
    
n_samples = args.jets
compare = True if (len(epochs) > 1 or len(setups) > 1 or len(wmets) > 1) else False

add_axis = True if args.add_axis == 'yes' else False
FGSM_setup = args.FGSM_setup
log_axis = True if args.log_axis == 'yes' else False
save_mode = args.save_mode


# run version with more complex styling for one requested model only
# (otherwise, get thresholds)
force_compare = True if args.force_compare == 'yes' else False

line_thickness = args.line_thickness

restrict = float(args.restrict)
restrict_text = f'_restrictedBy{restrict}' if restrict > 0 else '_restrictedByInf'
print('perturbation',restrict_text)

if save_mode=='AUC':
    epochs = range(epochs[0],epochs[1]+1)
    
print(f'Evaluate training at epoch {at_epoch}')
print(f'With weighting method {weighting_method}')

gamma = [((weighting_method.split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0] for weighting_method in wmets]
alphaparse = [((weighting_method.split('_gamma')[-1]).split('_alpha')[-1]).split('_adv_tr_eps')[0] for weighting_method in wmets]
epsilon = [(weighting_method.split('_adv_tr_eps')[-1]) for weighting_method in wmets]
print('gamma',gamma)
print('alpha',alphaparse)
print('epsilon',epsilon)
# debug setup
#sys.exit()

wm_def_text = build_wm_text_dict(gamma,alphaparse,epsilon)
    
'''

    Load inputs and targets
    
'''

test_input_file_paths = [preprocessed_path + f'test_inputs_%d.pt' % k for k in range(NUM_DATASETS)]
test_target_file_paths = [preprocessed_path + f'test_targets_%d.pt' % k for k in range(NUM_DATASETS)]



# note: the default case without parameters in the function input_indices_wanted returns all high-level variables, as well as 28 (all) features for the first 6 tracks
used_variables = input_indices_wanted()
slices = torch.LongTensor(used_variables)
# use n_input_features as the number of inputs to the model (later)
n_input_features = len(slices)


test_inputs = torch.cat(tuple(torch.load(ti)[:,slices] for ti in test_input_file_paths))
print('test inputs done')

test_targets = torch.cat(tuple(torch.load(ti) for ti in test_target_file_paths))
print('test targets done')

len_test = len(test_targets)
print('number of test inputs', len_test)

jetFlavour = test_targets+1


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
    

    

if compare == False and force_compare == False:
    with torch.no_grad():
        
        checkpoint = torch.load(f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{weighting_method}_{NUM_DATASETS}_{n_samples}/model_{prev_epochs}_epochs{weighting_method}_{NUM_DATASETS}_datasets_{n_samples}.pt', map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        predictions = model(test_inputs).detach().numpy()
        
        wm_text = wm_def_text[weighting_method]
        #'''
        fig = plt.figure(figsize=[12,12])       
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==0, torch.ones(len_test), torch.zeros(len_test)),predictions[:,0])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for b-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        
        plt.legend([f'Classifier: epoch {at_epoch}\n{wm_text}, AUC = {customauc:.4f}'],title='ROC b tagging',loc='lower right',fontsize=22,title_fontsize=24)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        #plt.title(f'ROC b tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files)')
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_b_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(eval_path + f'roc_curves/new_roc_b_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_b_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_b_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        
        fig = plt.figure(figsize=[12,12])
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==1, torch.ones(len_test), torch.zeros(len_test)),predictions[:,1])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for c-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        
        plt.legend([f'Classifier: epoch {at_epoch}\n{wm_text}, AUC = {customauc:.4f}'],title='ROC c tagging',loc='lower right',fontsize=22,title_fontsize=24)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        #plt.title(f'ROC c tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files)')
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_c_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(eval_path + f'roc_curves/new_roc_c_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_c_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_c_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)

        
        fig = plt.figure(figsize=[12,12])
        fpr,tpr,thresholds = metrics.roc_curve(torch.where(test_targets==2, torch.ones(len_test), torch.zeros(len_test)),predictions[:,2])
        plt.plot(fpr,tpr)
        customauc = metrics.auc(fpr,tpr)
        print(f"auc for udsg-tagging {wm_text}: {metrics.auc(fpr,tpr)}")
        
        plt.legend([f'Classifier: epoch {at_epoch}\n{wm_text}, AUC = {customauc:.4f}'],title='ROC udsg tagging',loc='lower right',fontsize=22,title_fontsize=24)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        #plt.title(f'ROC udsg tagging after {at_epoch} epochs,\nevaluated on {len_test} jets ({NUM_DATASETS} files)')
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_udsg_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)
        fig.savefig(eval_path + f'roc_curves/new_roc_udsg_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_udsg_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_udsg_tagging_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #'''


        '''
            B vs Light jets
        '''
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==3)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==3)]
        
        
        # now select only those that won't lead to division by zero
        # just to be safe: select only those values where the range is 0-1
        # because we slice based on the outputs, and have to apply the slicing in exactly the same way for targets and outputs, the targets need to go first (slicing with the 'old' outputs), then slice the outputs
        
        matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,2]) != 0]
        matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,2]) != 0]
        

        
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve((matching_targets==0),(matching_predictions[:,0])/(matching_predictions[:,0]+matching_predictions[:,2]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for B vs UDSG {wm_text}: {customauc}")
        
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc)],title='ROC B vs L',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(eval_path + f'roc_curves/new_roc_BvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_BvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_BvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        
        fig = plt.figure(figsize=[12,12],num=40)
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        
        plt.ylabel('TPR/FPR B vs L')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}'],title='B vs L',loc='center',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(eval_path + f'roc_curves/new_roc_BvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_BvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_BvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #sys.exit()
        '''
            B vs C jets
        '''
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==1) | (jetFlavour==2)]
        
        
        
        # now select only those that won't lead to division by zero
        #matching_inputs = matching_inputs[(1-matching_predictions[:,3]) != 0]
        matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]) != 0]
        matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]) != 0]

        #len_BvsC = len(matching_targets)
        
        

        
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve((matching_targets==0),(matching_predictions[:,0])/(matching_predictions[:,0]+matching_predictions[:,1]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for B vs C {wm_text}: {customauc}")
        
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for b vs. c\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc)],title='ROC B vs C',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvC_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)    
        fig.savefig(eval_path + f'roc_curves/new_roc_BvC_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_BvC_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_BvC_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        
        fig = plt.figure(figsize=[12,12],num=40)
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        #customauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG {wm_text}: {customauc}")
        
        plt.ylabel('TPR/FPR B vs C')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}'],title='B vs C',loc='lower left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_BvC_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(eval_path + f'roc_curves/new_roc_BvC_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_BvC_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_BvC_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #sys.exit()
        '''
            C vs B jets
        '''
        # inputs and targets are the same as for the previous classifier / discriminator
       
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve(matching_targets==1,(matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for C vs B {wm_text}: {customauc}")
        
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for c vs. b\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc)],title='ROC C vs B',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvB_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400) 
        fig.savefig(eval_path + f'roc_curves/new_roc_CvB_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_CvB_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_CvB_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)         
        
        fig = plt.figure(figsize=[12,12],num=40)
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        #customauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG {wm_text}: {customauc}")
        
        plt.ylabel('TPR/FPR C vs B')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}'],title='C vs B',loc='lower left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvB_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(eval_path + f'roc_curves/new_roc_CvB_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_CvB_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_CvB_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        #sys.exit() 
        '''
            C vs Light jets
        '''
        matching_targets = test_targets[(jetFlavour==2) | (jetFlavour==3)]
        #del test_targets
        
        matching_predictions = predictions[(jetFlavour==2) | (jetFlavour==3)]
        
        
        
        
       
        matching_targets = matching_targets[(matching_predictions[:,1]+matching_predictions[:,2]) != 0]
        matching_predictions = matching_predictions[(matching_predictions[:,1]+matching_predictions[:,2]) != 0]

        #len_CvsUDSG = len(matching_targets)


        
        fig = plt.figure(figsize=[12,12],num=40)
        fpr_custom,tpr_custom,thresholds_custom = metrics.roc_curve(matching_targets==1,(matching_predictions[:,1])/(matching_predictions[:,1]+matching_predictions[:,2]))
        plt.plot(tpr_custom,fpr_custom)
        customauc = metrics.auc(fpr_custom,tpr_custom)
        print(f"auc for C vs UDSG {wm_text}: {customauc}")
        
        plt.ylabel('mistag rate')
        plt.xlabel('efficiency')
        #plt.title(f'ROC for c vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(-0.05,1.05)
        plt.ylim(bottom=1e-3)
        plt.yscale('log')
        plt.legend([f'Classifier: {wm_text}\nepoch {at_epoch}, '+'AUC = {:.4f}'.format(customauc)],title='ROC C vs L',loc='upper left',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(eval_path + f'roc_curves/new_roc_CvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_CvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_CvL_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        
        
        fig = plt.figure(figsize=[12,12],num=40)
        plt.plot(thresholds_custom,tpr_custom,c='blue')
        plt.plot(thresholds_custom,fpr_custom,linestyle='dashed',c='blue')
        #customauc = metrics.auc(fpr,tpr)
        #print(f"auc for B vs UDSG {wm_text}: {customauc}")
        
        plt.ylabel('TPR/FPR C vs L')
        plt.xlabel('Threshold')
        #plt.title(f'ROC for b vs. udsg\n After {at_epoch} epochs, evaluated on {len_test} jets ({NUM_DATASETS} files)')
        #plt.xlim(-0.05,1.05)
        #plt.ylim(bottom=1e-3)
        plt.xlim((0,1))
        #plt.yscale('log')
        plt.legend([f'TPR Classifier: epoch {at_epoch}\n{wm_text}', f'FPR Classifier: epoch {at_epoch}\n{wm_text}'],title='C vs L',loc='upper right',fontsize=22,title_fontsize=24)
        plt.grid(which='minor', alpha=0.9)
        plt.grid(which='major', alpha=1, color='black')
        #fig.savefig(f'/home/um106329/aisafety/june_21/evaluate/roc_curves/new_roc_CvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}.png', bbox_inches='tight', dpi=400)  
        fig.savefig(eval_path + f'roc_curves/new_roc_CvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/new_roc_CvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/new_roc_CvL_thresholds_weighting_method{weighting_method}_at_epoch_{at_epoch}_{len_test}_jets_training_{NUM_DATASETS}_{n_samples}{restrict_text}.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
        sys.exit()
            
else:
    # ================================================================================
    #
    #
    #              Compare different things like epochs, methods, parameters
    #
    # ................................................................................
    #
    #                              Check what is requested
    #
    # ................................................................................
    # construct a combination of properties for every ROC
    # if only one thing varies, adjust the length of the other properties to match the max.
    # epochs and wmets defines the file that has to be read
    # setups controls how the inputs are constructed
    n_compare = max(len(epochs),len(setups),len(wmets))
    print(f'Compare {n_compare} setups.')
    if len(epochs) == 1: 
        same_epoch = True
        epochs = n_compare * epochs
    else:
        same_epoch = False

    if len(setups) == 1:
        same_setup = True
        setups = n_compare * setups
    else:
        same_setup = False

    if len(wmets) == 1: 
        same_wm = True
        wmets = n_compare * wmets
    else:
        same_wm = False

    # get the output variable or discriminator that will be compared
    outdisc = setups[0].split('_')[0]
    print(outdisc)

    # get linestyle / colour depending on what is requested
    # all raw, compare different epochs, all raw compare different weighting methods --> just different colours, linestyle identical
    # raw and distorted for different epochs, but same parameter --> raw - / distorted --
    # basic and adversarial training --> basic - / adversarial --
    # raw and distorted for same epoch, but different parameters --> raw - / distorted -, just different colours

    # evaluate the distorted inputs at the end, ckeck if it's always the same parameter / method to decide how the plots will look like (see different styles above)
    non_raw_setups_sigma = []
    non_raw_setups_epsilon = []
    for s in setups:
        if 'sigma' in s:
            non_raw_setups_sigma.append(s.split('sigma')[-1])
        elif 'eps' in s or 'sys' in s:
            non_raw_setups_epsilon.append(s.split('eps')[-1])

    if len(non_raw_setups_sigma) == 0 and len(non_raw_setups_epsilon) == 0:
        # everything is raw only, so the order does not matter --> always use undisturbed inputs
        raw_only = True
    elif len(non_raw_setups_sigma) == 0 and len(non_raw_setups_epsilon) != 0:
        # FGSM (or SYS), no Noise
        raw_only = False
        if non_raw_setups_epsilon.count(non_raw_setups_epsilon[0]) == len(non_raw_setups_epsilon):
            always_same_parameter = True
            that_one_epsilon = non_raw_setups_epsilon[0]
    elif len(non_raw_setups_sigma) != 0 and len(non_raw_setups_epsilon) == 0:
        # Noise, no FGSM / SYS
        raw_only = False
        if non_raw_setups_sigma.count(non_raw_setups_sigma[0]) == len(non_raw_setups_sigma):
            always_same_parameter = True
            that_one_sigma = non_raw_setups_sigma[0]
    else:
        # compare Noise, FGSM / SYS in same script
        raw_only = False

    has_basic = False
    has_adv = False
    for w in wmets:
        if 'adv' in w:
            has_adv = True
        else:
            has_basic = True
    has_nonalt = False
    has_alt = False
    for w in wmets:
        if 'alt' in w:
            has_alt = True
        if '_ptetaflavloss' in w:
            has_nonalt = True

    new_line_before_AUC = True

    possible_colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    n_comp_div10, n_comp_rem10 = divmod(n_compare,10)
    possible_colours = possible_colours*n_comp_div10 + possible_colours[:n_comp_rem10]
    #print(possible_colours)

    if raw_only:  # no attacks for comparison
        if same_wm:
            # just go through standard colours, just different epochs, no attacks, all solid lines
            used_colours = possible_colours[:n_compare]
            legend_setup = 'epochs'
            linestyles = ['-' for l in range(n_compare)]
            addition_leg_text = '\n'+wm_def_text[weighting_method]
            individual_legend = [f'Epoch {e}' for e in epochs]
            print('All raw, same weighting method, compare different epochs.')
            #print(linestyles)
            #print(addition_leg_text)
            #print(individual_legend)
            #print(used_colours)
            #sys.exit()
            new_line_before_AUC = False

        else:
            if has_adv != has_basic:  # XOR = means only one type of training shall be used (either basic or adversarial, not both)
                # everything solid lines only
                used_colours = possible_colours[:n_compare]
                linestyles = ['-' for l in range(n_compare)]
                print('All raw, but different weighting methods.')
                if has_alt and has_nonalt:
                    linestyles = []
                    used_colours = []
                    colourpointer = 0
                    for i,w in enumerate(wmets):
                        if (i>0) and (i%2 == 0):
                            colourpointer += 1
                        used_colours.append(possible_colours[colourpointer])
                        if 'alt' in w:
                            linestyles.append('--')
                        else:
                            linestyles.append('-')
                new_line_before_AUC = False
            else:
                # basic and adversarial training present in selection
                # adversarial shall get dashed lines, but colour shall correspond to basic training colour
                # assuming even number of weighting methods, basic and adversarial always consecutive (and corresponding)
                if n_compare%2 != 0:
                    print('Check number of weighting methods, need basic / adversarial consecutively')
                    sys.exit()
                print('All raw, but compare basic with adversarial training.')
                linestyles = []
                used_colours = []
                colourpointer = 0
                for i,w in enumerate(wmets):
                    if (i>0) and (i%2 == 0):
                        colourpointer += 1
                    used_colours.append(possible_colours[colourpointer])
                    if 'adv' in w:
                        linestyles.append('--')
                    else:
                        linestyles.append('-')
                #print(linestyles)
                #print(used_colours)

            if same_epoch:
                legend_setup = 'wmets'
                addition_leg_text = '\n'+f'Epoch {at_epoch}'
                individual_legend = [wm_def_text[w] for w in wmets]
            else:
                legend_setup = 'wmets_epochs'
                addition_leg_text = ''
                individual_legend = [wm_def_text[w]+f'\nEpoch {e}' for w,e in zip(wmets,epochs)]
            #print(addition_leg_text)
            #print(individual_legend)
            #sys.exit()
    else:
        linestyles = []
        used_colours = []
        individual_legend = []
        colourpointer = 0
        n_raw = 0
        for i,s in enumerate(setups):
            if (i>0) and (i%2 == 0):
                colourpointer += 1
            used_colours.append(possible_colours[colourpointer])
            if ('sigma' in s) or ('sys' in s) or ('eps' in s):
                linestyles.append('--')
                if 'sigma' in s:
                    sig = s.split('sigma')[-1]
                    individual_legend.append(f'Noise $\sigma=${sig}')
                elif 'sys' in s:
                    eps = s.split('sys')[-1]
                    uptext = '+' if 'UP' in s else '-'
                    individual_legend.append(f'SYS {uptext}$\epsilon=${eps}')
                else:
                    eps = s.split('eps')[-1]
                    individual_legend.append(f'FGSM $\epsilon=${eps}')

            else:
                linestyles.append('-')
                individual_legend.append('Raw')
                n_raw += 1
        addition_leg_text = ''
        if same_epoch:
            # how it was done previously
            #addition_leg_text = addition_leg_text + f', Epoch {at_epoch}'
            # for paper
            #addition_leg_text = addition_leg_text # which is equivalent to not setting it at all
            # new MLPhysics
            new_line_before_AUC = False
        else:
            for e in range(n_compare):
                # more info for us
                #individual_legend[e] = f'Epoch {epochs[e]}, ' + individual_legend[e]
                # less explicit for paper; no mention of epoch here
                #individual_legend[e] = individual_legend[e]
                continue
            new_line_before_AUC = False
        if same_wm:
            addition_leg_text = addition_leg_text + '\n'+wm_def_text[weighting_method]
            if n_raw == 1:
                used_colours = possible_colours
        else:
            for w in range(n_compare):
                # new MLPhysics
                individual_legend[w] = f'{wm_def_text[wmets[w]]}\n' + individual_legend[w]


    if new_line_before_AUC:
        for n in range(n_compare):
            individual_legend[n] = individual_legend[n]+',\n'

    else:
        for n in range(n_compare):
            individual_legend[n] = individual_legend[n]+', '


    #print(setups)
    #print(used_colours)
    #print(individual_legend)
    #sys.exit()
    # ................................................................................
    #
    #                              Prepare inputs, targets
    #
    # ................................................................................


    # prepare inputs to be able to calculate discriminators
    if outdisc in ['b','c','udsg']:
        # simply test inputs / targets
        matching_inputs = test_inputs
        del test_inputs
        gc.collect()
        matching_targets = test_targets
        del test_targets
        gc.collect()

    elif outdisc == 'BvL':
        # create BvL inputs / targets
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==3)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==3)]
        del test_inputs
        gc.collect()


    elif outdisc == 'BvC':
        # create BvC inputs / targets
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2)]
        del test_inputs
        gc.collect()


    elif outdisc == 'CvB':
        # create CvB inputs / targets
        matching_targets = test_targets[(jetFlavour==1) | (jetFlavour==2)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==1) | (jetFlavour==2)]
        del test_inputs
        gc.collect()


    elif outdisc == 'CvL':
        # create CvL inputs / targets
        matching_targets = test_targets[(jetFlavour==2) | (jetFlavour==3)]
        del test_targets
        gc.collect()

        matching_inputs = test_inputs[(jetFlavour==2) | (jetFlavour==3)]
        del test_inputs
        gc.collect()

        
    del jetFlavour
    gc.collect()
    print('Prepared matching inputs and targets for this output/discriminator.')



    # .........................................................................................................
    #                                        Begin figure and style
    # .........................................................................................................
    #fig = plt.figure(figsize=[12,12])
    fig,ax = plt.subplots(figsize=[12,12])
    #ax.set_xlim(left=-0.05,right=1.05)
    ax.set_xlim(left=0.,right=1.)

    if log_axis:
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-3)
        ax.set_ylim(top=1)

    if outdisc=='b':
        tag_name = ' (b)'
        mistag_name = ' (c,udsg)'
    elif outdisc=='c':
        tag_name = ' (c)'
        mistag_name = ' (b,udsg)'
    elif outdisc=='udsg':
        tag_name = ' (udsg)'
        mistag_name = ' (b,c)'
    else:
        if outdisc[0]=='B':
            tag_name = ' (b)'
        elif outdisc[0]=='C':
            tag_name = ' (c)'
        if outdisc[-1]=='B':
            mistag_name = ' (b)'
        elif outdisc[-1]=='C':
            mistag_name = ' (c)'
        elif outdisc[-1]=='L':
            mistag_name = ' (udsg)'
    if log_axis:
        ax.set_ylabel('Mistagging rate'+mistag_name)
        ax.set_xlabel('Tagging efficiency'+tag_name)
    else:
        ax.set_xlabel('Mistagging rate'+mistag_name)
        ax.set_ylabel('Tagging efficiency'+tag_name)
    if add_axis:
        if log_axis:
            if outdisc=='b':
                ax.plot([0.47,0.57,0.57,0.47,0.47],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([1,0.57],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.565,0.47],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.565, .15, .3, .15])
                ax2.set_xlim(0.47,0.57)
            elif outdisc=='c':
                ax.plot([0.06,0.11,0.11,0.06,0.06],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.55,0.06],[3e-1,2e-2],'--',color='grey')
                ax.plot([0.975,0.11],[8e-2,8e-3],'--',color='grey')
                ax2 = plt.axes([.55, .6, .3, .15])
                ax2.set_xlim(0.06,0.11)
            elif outdisc=='udsg':
                ax.plot([0.04,0.09,0.09,0.04,0.04],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.04],[8.4e-1,2e-2],'--',color='grey')
                ax.plot([0.512,0.09],[2.2e-1,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .71, .3, .15])
                ax2.set_xlim(0.04,0.09)
            elif outdisc=='BvL':
                ax.plot([0.6,0.75,0.75,0.6,0.6],[8e-3,8e-3,1.2e-2,1.2e-2,8e-3],'--',color='black')
                #ax.plot([0.125,0.65],[5.5e-3,1.2e-2],'--',color='grey')
                #ax.plot([0.435,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .35, .3, .15])
                ax2.set_xlim(0.6,0.75)
            elif outdisc=='BvC':
                ax.plot([0.35,0.45,0.45,0.35,0.35],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([1,0.45],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.565,0.35],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.565, .15, .3, .15])
                ax2.set_xlim(0.35,0.45)
            elif outdisc=='CvB':
                ax.plot([0.05,0.15,0.15,0.05,0.05],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.05],[8.4e-1,2e-2],'--',color='grey')
                ax.plot([0.512,0.15],[2.2e-1,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .71, .3, .15])
                ax2.set_xlim(0.05,0.15)
            elif outdisc=='CvL':
                ax.plot([0.1,0.25,0.25,0.1,0.1],[8e-3,8e-3,1.2e-2,1.2e-2,8e-3],'--',color='black')
                #ax.plot([0.89,0.91,0.91,0.89,0.89],[7e-1,7e-1,9e-1,9e-1,7e-1],'--',color='black')
                #ax.plot([0.75,0.85,0.85,0.75,0.85],[5e-1,5e-1,96-1,6e-1,5e-1],'--',color='black')
                
                #ax.plot([0.6,0.1],[1.2e-1,8e-3],'--',color='grey')
                
                #ax.plot([0.43,0.75],[8.2e-1,6e-1],'--',color='grey')
                
                #ax.plot([0.94,0.25],[2.2e-2,8e-3],'--',color='grey')
                
                #ax.plot([0.43,0.75],[2.2e-1,5e-1],'--',color='grey')
                #ax2 = plt.axes([.22, .71, .28, .15])
                ax2 = plt.axes([.54, .48, .31, .18])
                ax2.set_xlim(0.1,0.25)   
                #ax2.set_xlim(0.75,0.85)   
            ax2.set_yscale('log')
            if outdisc not in ['BvL','CvL']:
                ax2.set_ylim(8e-3,2e-2)
            else:
                ax2.set_ylim(8e-3,1.2e-2)
                #ax2.set_ylim(5e-1,6e-1)
        else:  # ToDo! But: not urgent
            if outdisc=='b':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='c':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='udsg':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='BvL':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='BvC':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='CvB':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            elif outdisc=='CvL':
                ax.plot([0.65,0.75,0.75,0.65,0.65],[8e-3,8e-3,2e-2,2e-2,8e-3],'--',color='black')
                ax.plot([0.085,0.65],[5.5e-3,2e-2],'--',color='grey')
                ax.plot([0.515,0.75],[1.5e-3,8e-3],'--',color='grey')
                ax2 = plt.axes([.22, .15, .3, .15])
                ax2.set_xlim(0.65,0.75)
            ax2.set_ylim(-0.05,1.05)


    # --------------------------------------------------------------------------------
    #
    #                        Run over all requested combinations
    #
    # ................................................................................
    if save_mode == 'AUC':
        auc_list =  []
    for i in range(n_compare):
        # https://pytorch.org/docs/stable/generated/torch.autograd.set_grad_enabled.html#torch.autograd.set_grad_enabled --> activate gradients based on a condition (avoiding exec statements --> make code cleaner without duplication)
        activate_grad = True if 'eps' in setups[i] else False
        with torch.set_grad_enabled(activate_grad):  # if FGSM attack is used --> initially, require grad, only disable for final step in external function
            # load correct checkpoint
            if (i == 0) or (same_epoch == False) or (same_wm == False):
                checkpoint = torch.load(f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{wmets[i]}_{NUM_DATASETS}_{n_samples}/model_{epochs[i]}_epochs{wmets[i]}_{NUM_DATASETS}_datasets_{n_samples}.pt', map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                if ('_ptetaflavloss' in wmets[i]) or ('_altptetaflavloss' in wmets[i]):
                    if 'focalloss' not in wmets[i]:
                        criterion = nn.CrossEntropyLoss(reduction='none')
                    elif 'focalloss' in wmets[i]:
                        if 'alpha' not in wmets[i]:
                            alpha = None
                        else:
                            commasep_alpha = [a for a in ((wmets[i].split('_alpha')[-1]).split('_adv_tr_eps')[0]).split(',')]
                            alpha = torch.Tensor([float(commasep_alpha[0]),float(commasep_alpha[1]),float(commasep_alpha[2])]).to(device)
                        if 'gamma' not in wmets[i]:
                            gamma = 2.0
                        else:
                            gamma = float(((wmets[i].split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0])
                        criterion = FocalLoss(alpha, gamma, reduction='none')

                else:
                    criterion = nn.CrossEntropyLoss()

            # .........................................................................................................
            #                                             predict
            # .........................................................................................................
            
            fgsm_setup_text = ''
            # use raw or distorted
            if 'raw' in setups[i]:
                matching_predictions = model(matching_inputs).detach().numpy()
                #if ('adv' in wmets[i]) and has_basic:
                #    this_line = '--'
                #else:
                #    this_line = '-'
                #setup_text = ''

            elif 'sigma' in setups[i]:
                sig = float(setups[i].split('sigma')[-1])
                matching_predictions = model(apply_noise(matching_inputs,sig,filtered_indices=used_variables,restrict_impact=restrict)).detach().numpy()
                #this_line = '--'
                #setup_text = f'Noise $\sigma={sig}$'   
            elif 'sys' in setups[i]:
                eps = float(setups[i].split('sys')[-1])
                up = True if 'UP' in setups[i] else False
                matching_predictions = model(syst_var(eps,matching_inputs,filtered_indices=used_variables,restrict_impact=restrict,up=up)).detach().numpy()
                #this_line = '--'
                #setup_text = f'SYS $\epsilon={eps}$'   
            elif 'eps' in setups[i]:
                eps = float(setups[i].split('eps')[-1])
                #matching_inputs.requires_grad = True
                if FGSM_setup == '-1':
                    # the most individual case is here, for each model and each epoch, craft the "worst-case" FGSM samples, i.e. FGSM-model == eval-model
                    if 'adv' in wmets[i]:
                        linestyles[i] = '-.'
                    else:
                        linestyles[i] = '--'
                    matching_predictions = model(fgsm_attack(eps,matching_inputs,matching_targets,model,criterion,dev=device,filtered_indices=used_variables,restrict_impact=restrict)).detach().numpy()
                else:
                    # less individual FGSM, i.e. need at least some different model from which FGSM is generated,
                    fgsm_model = nn.Sequential(nn.Linear(n_input_features, 100),
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
    
                    fgsm_model.to(device)
                    fgsm_model.eval()
            
            
                    if 'At' in FGSM_setup:
                        epoch_for_FGSM = FGSM_setup.split('At')[-1]
                        # fixed epoch
                        if 'Basic' in FGSM_setup:
                            # fixed epoch, and fixed model = Basic training
                            wm_for_FGSM = '_ptetaflavloss_focalloss_gamma25.0'
                        elif 'Adversarial' in FGSM_setup:
                            # fixed epoch, and fixed model = Adversarial training
                            wm_for_FGSM = '_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01'
                        elif 'Extra' in FGSM_setup:
                            # fixed epoch, and fixed model = some extra training, independent of the two, ToDo
                            sys.exit()
                        else:
                            # fixed epoch, but for each model / training independently
                            wm_for_FGSM = wmets[i]
                        
                            
                    else:
                        epoch_for_FGSM = epochs[i]
                        if 'Basic' in FGSM_setup:
                            # varying epochs, but only from Basic training
                            wm_for_FGSM = '_ptetaflavloss_focalloss_gamma25.0'
                        elif 'Adversarial' in FGSM_setup:
                            # varying epochs, but only from Adversarial training
                            wm_for_FGSM = '_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01'
                        else:
                            # varying epochs, with fixed Extra model / training, ToDo
                            # Note: varying epochs, individual per model / training, is already handled with the -1 case
                            sys.exit()
            
                    if 'Basic' in FGSM_setup:
                        linestyles[i] = '--'
                    else:
                        linestyles[i] = '-.'
            
                    # could write additional info to plot that captures the setup (individual FGSM, or one attacked sample for all)
                    fgsm_setup_text = f'* FGSM attack based on epoch {epoch_for_FGSM} with\n{wm_def_text[wm_for_FGSM]}'
                    # only use it in an abbreviated way for the filename
                    if i == 0:
                        print('FGSM setup:', FGSM_setup)
                    
                    fgsm_checkpoint = torch.load(f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{wm_for_FGSM}_{NUM_DATASETS}_{n_samples}/model_{epoch_for_FGSM}_epochs{wm_for_FGSM}_{NUM_DATASETS}_datasets_{n_samples}.pt', map_location=device)
                    fgsm_model.load_state_dict(fgsm_checkpoint["model_state_dict"])
                    matching_predictions = model(fgsm_attack(eps,matching_inputs,matching_targets,fgsm_model,criterion,dev=device,filtered_indices=used_variables,restrict_impact=restrict)).detach().numpy()
                #this_line = '--'      
                #setup_text = f'FGSM $\epsilon={eps}$'   
            #matching_predictions = np.float32(matching_predictions)                                 
            print('Predictions done.')

        wm_text = wm_def_text[wmets[i]]

        #this_label = wm_text + '\n' + setup_text

        this_line = linestyles[i]
        this_colour = used_colours[i]
        this_legtext = individual_legend[i]
        
        if compare == False and force_compare == True:
            this_colour = 'red'

        # .........................................................................................................
        #                                             ROC & AUC
        # .........................................................................................................
        if outdisc == 'b':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==0, torch.ones(len_test), torch.zeros(len_test)),matching_predictions[:,0])
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'upper left'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            print(f"auc for b-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)
            legtitle = 'ROC b tagging'+addition_leg_text
        elif outdisc == 'c':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==1, torch.ones(len_test), torch.zeros(len_test)),matching_predictions[:,1])
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            print(f"auc for c-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)
            legtitle = 'ROC c tagging'+addition_leg_text
        elif outdisc == 'udsg':
            fpr,tpr,_ = metrics.roc_curve(torch.where(matching_targets==2, torch.ones(len_test), torch.zeros(len_test)),matching_predictions[:,2])
            customauc = metrics.auc(fpr,tpr)
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)
            print(f"auc for udsg-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC udsg tagging'+addition_leg_text
        # every discriminator has different properties / different conditions for the computation to work
        elif outdisc == 'BvL':
            # checking the predictions works only for every iteration specifically (because this depends on the model with which predictions are done)
            # now select only those that won't lead to division by zero
            # just to be safe: select only those values where the range is 0-1
            # because we slice based on the outputs, and have to apply the slicing in exactly the same way for targets and outputs, the targets need to go first (slicing with the 'old' outputs), then slice the outputs
            actually_matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,2]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,2]) != 0]

            fpr,tpr,_ = metrics.roc_curve((actually_matching_targets==0),(matching_predictions[:,0])/(matching_predictions[:,0]+matching_predictions[:,2]))
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'upper left'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)
            print(f"auc for bvl-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC B vs L'+addition_leg_text
            del actually_matching_targets
            gc.collect()

        elif outdisc == 'BvC':
            actually_matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]) != 0]

            fpr,tpr,_ = metrics.roc_curve((actually_matching_targets==0),(matching_predictions[:,0])/(matching_predictions[:,0]+matching_predictions[:,1]))
            del matching_predictions
            gc.collect()
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'upper left'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)
            print(f"auc for bvc-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC B vs C'+addition_leg_text
            del actually_matching_targets
            gc.collect()

        elif outdisc == 'CvB':
            actually_matching_targets = matching_targets[(matching_predictions[:,0]+matching_predictions[:,1]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,0]+matching_predictions[:,1]) != 0]

            fpr,tpr,_ = metrics.roc_curve(actually_matching_targets==1,(matching_predictions[:,1])/(matching_predictions[:,0]+matching_predictions[:,1]))
            del matching_predictions
            gc.collect()
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)
            print(f"auc for cvb-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC C vs B'+addition_leg_text
            del actually_matching_targets
            gc.collect()

        elif outdisc == 'CvL':
            actually_matching_targets = matching_targets[(matching_predictions[:,1]+matching_predictions[:,2]) != 0]
            matching_predictions = matching_predictions[(matching_predictions[:,1]+matching_predictions[:,2]) != 0]

            fpr,tpr,_ = metrics.roc_curve(actually_matching_targets==1,(matching_predictions[:,1])/(matching_predictions[:,1]+matching_predictions[:,2]))
            del matching_predictions
            gc.collect()
            customauc = metrics.auc(fpr,tpr)
            if save_mode=='AUC':
                auc_list.append(customauc)
                continue
            if log_axis:
                legloc = 'lower right'
                ax.plot(tpr,fpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            else:
                legloc = 'lower right'
                ax.plot(fpr,tpr,label=f'{this_legtext}AUC = {customauc:.3f}', linestyle=this_line, color=this_colour,linewidth=line_thickness)
            if add_axis:
                if log_axis:
                    ax2.plot(tpr,fpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)                        
                else:
                    ax2.plot(fpr,tpr, linestyle=this_line, color=this_colour,linewidth=line_thickness)
            print(f"auc for cvl-tagging epoch {epochs[i]} {wm_text}, setup {setups[i]}: {customauc}")
            legtitle = 'ROC C vs L'+addition_leg_text
            del actually_matching_targets
            gc.collect()
            
    
    del fpr
    del tpr
    gc.collect()

    
    if save_mode=='ROC':
        # .........................................................................................................
        #                                             style some more & save
        # .........................................................................................................
        leg = ax.legend(title=legtitle,loc=legloc,fontsize=20,title_fontsize=23,labelspacing=0.7)
        if 'right' in legloc:
            aligned = 'right'
        else:
            aligned = 'left'
        leg._legend_box.align = aligned
        ax.grid(which='minor', alpha=0.85)
        ax.grid(which='major', alpha=0.95, color='black')
        if add_axis:
            ax2.grid(which='minor', alpha=0.85)
            ax2.grid(which='major', alpha=0.95, color='black')

        #if FGSM_setup != -1:
        # don't need even more text on plots in paper, so just not plot the fgsm_setup
        if False:
            if log_axis:
                if add_axis == False:
                    ax.text(0,1.5e-3,fgsm_setup_text, fontsize=16)
                else:
                    ax.text(0,1e-2,fgsm_setup_text, fontsize=16)
            else:
                ax.text(0,0,fgsm_setup_text, fontsize=16)

        
        if log_axis:
            log_axis_text = '_logAx'
        else:
            log_axis_text = ''
        if add_axis:
            add_axis_text = '_addAx'
        else:
            add_axis_text = ''
        weighting_method = list(dict.fromkeys(wmets))
        at_epoch = list(dict.fromkeys(epochs))
        compare_setup = list(dict.fromkeys(setups))
        fgsm_setup_text = '' if fgsm_setup_text=='' else f'_FGSM_{FGSM_setup}'
        
        fig.savefig(eval_path + f'roc_curves/compare_new/roc_{outdisc}/{weighting_method}_e{at_epoch}_s{compare_setup}{log_axis_text}{add_axis_text}{fgsm_setup_text}_{len_test}_{NUM_DATASETS}_{n_samples}{restrict_text}_v2.png', bbox_inches='tight',dpi=600)
        fig.savefig(eval_path + f'roc_curves/compare_new/roc_{outdisc}/{weighting_method}_e{at_epoch}_s{compare_setup}{log_axis_text}{add_axis_text}{fgsm_setup_text}_{len_test}_{NUM_DATASETS}_{n_samples}{restrict_text}_v2.svg', bbox_inches='tight')
        fig.savefig(eval_path + f'roc_curves/compare_new/roc_{outdisc}/{weighting_method}_e{at_epoch}_s{compare_setup}{log_axis_text}{add_axis_text}{fgsm_setup_text}_{len_test}_{NUM_DATASETS}_{n_samples}{restrict_text}_v2.pdf', bbox_inches='tight')
        plt.show(block=False)
        time.sleep(5)
        plt.close('all')
        gc.collect(2)
    else:
        auc_array = np.array(auc_list)
        weighting_method = list(dict.fromkeys(wmets))
        #at_epoch = list(dict.fromkeys(epochs))
        compare_setup = list(dict.fromkeys(setups))
        ffgsm_setup_text = '' if fgsm_setup_text=='' else f'_FGSM_{FGSM_setup}.'
        np.save(eval_path + f'auc/{outdisc}/{weighting_method}_e{at_epoch}_s{compare_setup}{fgsm_setup_text}_{len_test}_{NUM_DATASETS}_{n_samples}{restrict_text}.npy',auc_array)
