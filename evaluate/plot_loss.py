import torch

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT, hep.style.fira, hep.style.firamath])

import argparse

import numpy as np

import sys
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/helpers/")
from tools import preprocessed_path, eval_path, FALLBACK_NUM_DATASETS, build_wm_text_dict

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument("files", type=int, help="Number of files for training")
parser.add_argument("wm", help="Weighting method: _noweighting, _ptetaflavloss, _altptetaflavloss or with additional _focalloss; specifying multiple weighting methods is possible (split by +)")
parser.add_argument("jets", help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type _-1 (using multiple: split them by , like so: _-1,_-1,_-1)")
args = parser.parse_args()

# example: !python plot_loss.py 230 _ptetaflavloss_focalloss_gamma25.0+_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01 _-1,_-1


NUM_DATASETS = args.files
NUM_DATASETS = FALLBACK_NUM_DATASETS if NUM_DATASETS < 0 else NUM_DATASETS
weighting_method = args.wm
wmets = [w for w in weighting_method.split('+')]

n_samples = args.jets
all_n_samples = [n[1:] for n in n_samples.split(',')]
print(all_n_samples)

gamma = [((weighting_method.split('_gamma')[-1]).split('_alpha')[0]).split('_adv_tr_eps')[0] for weighting_method in wmets]
alphaparse = [((weighting_method.split('_gamma')[-1]).split('_alpha')[-1]).split('_adv_tr_eps')[0] for weighting_method in wmets]
epsilon = [(weighting_method.split('_adv_tr_eps')[-1]) for weighting_method in wmets]
print('gamma',gamma)
print('alpha',alphaparse)
print('epsilon',epsilon)

wm_def_text = build_wm_text_dict(gamma,alphaparse,epsilon)

colorcode = ['darkblue', 'royalblue', 'forestgreen', 'limegreen', 'maroon','red','darkolivegreen','yellow', 'darkcyan', 'cyan']

wm_epochs_so_far = {
    '_ptetaflavloss_focalloss_gamma25.0' : 497,
    '_ptetaflavloss_focalloss' : 250, # placeholder
    '_ptetaflavloss_focalloss_gamma13.0_adv_tr_eps0.005' : 15, # placeholder
    '_ptetaflavloss_focalloss_gamma13.0' : 15, # placeholder
    '_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.005' : 47, # placeholder
    '_ptetaflavloss_focalloss_gamma25.0_adv_tr_eps0.01' : 1000,
    '_ptetaflavloss' : 198, # placeholder
    
}

plt.ioff()    
    
plt.figure(1,figsize=[13,10])

for k,wm in enumerate(wmets):
    
    all_tr_losses = []
    all_val_losses = []
        
    for i in range(1, wm_epochs_so_far[wm]+1):
        checkpoint = torch.load(f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{wm}_{NUM_DATASETS}_{all_n_samples[k]}/model_{i}_epochs{wm}_{NUM_DATASETS}_datasets_{all_n_samples[k]}.pt', map_location=torch.device(device))
        all_tr_losses.append(checkpoint['loss'])
        all_val_losses.append(checkpoint['val_loss'])
        
    all_epochs = np.arange(1,wm_epochs_so_far[wm]+1)  
    plt.plot(all_epochs, all_tr_losses,color=colorcode[k*2],label=f'Training loss ({wm_def_text[wm]})')
    plt.plot(all_epochs, all_val_losses,color=colorcode[k*2+1],label=f'Validation loss ({wm_def_text[wm]})')
        
plt.title(f"Training history", y=1.02)
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.legend()
# https://stackoverflow.com/a/9707180
leg = plt.legend()
# set the linewidth of each legend object
for legobj in leg.legendHandles:
    legobj.set_linewidth(3.0)
prev_epochs = [wm_epochs_so_far[wm] for wm in wmets]
plt.savefig(eval_path + f'loss_plots/{NUM_DATASETS}_files_{prev_epochs}_epochs_train_history_wm{wmets}_{all_n_samples}_samples_v2.png', bbox_inches='tight', dpi=400, facecolor='w', transparent=False)
plt.savefig(eval_path + f'loss_plots/{NUM_DATASETS}_files_{prev_epochs}_epochs_train_history_wm{wmets}_{all_n_samples}_samples_v2.svg', bbox_inches='tight')
plt.savefig(eval_path + f'loss_plots/{NUM_DATASETS}_files_{prev_epochs}_epochs_train_history_wm{wmets}_{all_n_samples}_samples_v2.pdf', bbox_inches='tight')
