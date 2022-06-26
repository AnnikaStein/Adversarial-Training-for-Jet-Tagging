import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader


import time
import random
import gc

import argparse

import os, sys


from fast_tensor_data_loader import FastTensorDataLoader

from focal_loss import FocalLoss, focal_loss
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/attack/")
from disturb_inputs import fgsm_attack

# depending on what's available, or force cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

if device == torch.device("cuda:0"):
    print('Do training on GPU0')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
sys.path.append("/home/um106329/aisafety/jet_flavor_MLPhysics/helpers/")
from tools import preprocessed_path, FALLBACK_NUM_DATASETS
from variables import input_indices_wanted
    
    
    
start = time.time()

# example: python training.py 230 0 1 _ptetaflavloss -1 yes no 0 equal equal equal -1 -1

parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument("files", type=int, help="Number of files for training (all files: if you don't know the total number of files, you can use -1, which currently stands for 230)")
parser.add_argument("prevep", type=int, help="Number of previously trained epochs")
parser.add_argument("addep", type=int, help="Number of additional epochs for this training")
parser.add_argument("wm", help="Weighting method, can be uniform binning (_ptetaflavloss) or custom (_altptetaflavloss)")
parser.add_argument("jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1")
parser.add_argument("dofastdl", help="Use fast DataLoader")
parser.add_argument("dofl", help="Use Focal Loss")
parser.add_argument("gamma", type=float, help="Gamma (exponent for focal loss)")
parser.add_argument("alpha1", help="Alpha (prefactor for focal loss) for category 1 or type 'equal' for no additional weights.")
parser.add_argument("alpha2", help="Alpha (prefactor for focal loss) for category 2 or type 'equal' for no additional weights.")
parser.add_argument("alpha3", help="Alpha (prefactor for focal loss) for category 3 or type 'equal' for no additional weights.")
parser.add_argument("epsilon", type=float, help="Do Adversarial Training with epsilon > 0, or put -1 to do basic training only.")
parser.add_argument("restrict", help="Restrict impact of the attack ? -1 for no, some positive value for yes")
args = parser.parse_args()

NUM_DATASETS = args.files
NUM_DATASETS = FALLBACK_NUM_DATASETS if NUM_DATASETS < 0 else NUM_DATASETS
prev_epochs = args.prevep
epochs = args.addep
weighting_method = args.wm
n_samples = args.jets
do_fastdataloader = args.dofastdl
do_FL = args.dofl
epsilon = args.epsilon

restrict = float(args.restrict)
restrict_text = f'_restrictedBy{restrict}' if restrict > 0 else '_restrictedByInf'

if do_FL == 'yes':
    fl_text = '_focalloss'
    
    gamma = args.gamma
    alphaparse1 = args.alpha1
    alphaparse2 = args.alpha2
    alphaparse3 = args.alpha3
            
    if gamma != 2.:
        fl_text += f'_gamma{gamma}'
    else:
        gamma = 2
     
    if alphaparse1 != 'equal':  # if no additional weights are specified in the submission, all three weights will be 'equal', so also the first one
        fl_text += f'_alpha{alphaparse1},{alphaparse2},{alphaparse3}'
        alpha = torch.Tensor([float(alphaparse1),float(alphaparse2),float(alphaparse3)]).to(device)
    else:
        alpha = None  # otherwise don't modify the additional weights
        
else:
    fl_text = ''

    
if epsilon > 0:
    at_text = f'_adv_tr_eps{epsilon}'
    print('FGSM',restrict_text)
else:
    at_text = ''
    # just to make sure the description does not contain a non-empty string for (non-)restricted, if there is no adversarial training in the first place
    # e.g. if nominal training is chosen, it does not matter what is written inside the restricted variable, it will not be stored anywhere in the file path
    restrict_text = ''
    
'''
    Available weighting methods:
        '_noweighting' :  apply no weighting factors at all
        '_ptetaflavloss' : balance flavour, and reweighted in pt & eta such that shapes per flavour are equal (absolute), uniform binning -- uses sample weights that will be multiplied with the loss
        '_altptetaflavloss' : almost like the previous one, but with custom binning -- uses sample weights that will be multiplied with the loss
'''

print(f'weighting method: {weighting_method}{fl_text}{at_text}{restrict_text}')   

# for the initial setup, reduce sources of randomness (e.g. if different methods will be used, they should start with the same initial weights), but later, using deterministic algs etc. would just slow things down without providing any benefit
if prev_epochs == 0:
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    parent_dir_1 = '/home/um106329/aisafety/jet_flavor_MLPhysics/saved_models'
    directory_1 = f'{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_{n_samples}'
    path_1 = os.path.join(parent_dir_1, directory_1)
    if not os.path.exists(path_1):
        os.mkdir(path_1)
        
    parent_dir_2 = '/hpcwork/um106329/jet_flavor_MLPhysics/saved_models'
    directory_2 = f'{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_{n_samples}'
    path_2 = os.path.join(parent_dir_2, directory_2)
    if not os.path.exists(path_2):
        os.mkdir(path_2)
        

bsize = 65536 # 2^16
lrate = 0.0001 # initial learning rate, only for first epoch



print(f'starting to train the model after {prev_epochs} epochs that were already done')
print(f'learning rate for this script: {lrate}')
print(f'batch size for this script: {bsize}')
    

with open(f"/home/um106329/aisafety/jet_flavor_MLPhysics/status_logfiles/logfile{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_files_{n_samples}_jets.txt", "a") as log:
    log.write(f"Setup: weighting method {weighting_method}{fl_text}{at_text}{restrict_text}, so far {prev_epochs} epochs done. Use lr={lrate} and bsize={bsize}. {n_samples} jets (-1 stands for all jets).\n")

    
train_input_file_paths = [preprocessed_path + f'train_inputs_%d.pt' % k for k in range(NUM_DATASETS)]
train_target_file_paths = [preprocessed_path + f'train_targets_%d.pt' % k for k in range(NUM_DATASETS)]
val_input_file_paths = [preprocessed_path + f'val_inputs_%d.pt' % k for k in range(NUM_DATASETS)]
val_target_file_paths = [preprocessed_path + f'val_targets_%d.pt' % k for k in range(NUM_DATASETS)]

if weighting_method == '_ptetaflavloss':
    train_sample_weights_file_paths = [preprocessed_path + f'train_sample_weights_%d.npy' % k for k in range(NUM_DATASETS)]
if weighting_method == '_altptetaflavloss':
    train_sample_weights_file_paths = [preprocessed_path + f'train_sample_weights_alt_%d.npy' % k for k in range(NUM_DATASETS)]
        

##### LOAD TRAINING SAMPLES #####
# note: the default case without parameters in the function input_indices_wanted returns all high-level variables, as well as 28 (all) features for the first 6 tracks
used_variables = input_indices_wanted()
slices = torch.LongTensor(used_variables)
# use n_input_features as the number of inputs to the model (later)
n_input_features = len(slices)

pre = time.time()
if weighting_method == '_ptetaflavloss' or weighting_method == '_altptetaflavloss':
    # if the loss shall be multiplied with sample weights after the calculation, one needs to add these as an additional column to the dataset inputs (otherwise the indices would not match up when using the dataloader)
    # adapted from https://stackoverflow.com/a/66375624/14558181
    if do_fastdataloader == 'no':
        allin = ConcatDataset([TensorDataset(torch.cat((torch.load(train_input_file_paths[f])[:,slices], torch.from_numpy(np.load(train_sample_weights_file_paths[f])).to(torch.float32).unsqueeze(1)), dim=1) , torch.load(train_target_file_paths[f])) for f in range(NUM_DATASETS)])
    else:
        global train_inputs
        global train_targets
        train_inputs = torch.cat([torch.cat((torch.load(train_input_file_paths[f])[:,slices], torch.from_numpy(np.load(train_sample_weights_file_paths[f])).to(torch.float32).unsqueeze(1)), dim=1) for f in range(NUM_DATASETS)])
        train_targets = torch.cat([torch.load(path) for path in train_target_file_paths])
        
else:
    if do_fastdataloader == 'no':
        allin = ConcatDataset([TensorDataset(torch.load(train_input_file_paths[f])[:,slices], torch.load(train_target_file_paths[f])) for f in range(NUM_DATASETS)])
    else:
        train_inputs = torch.cat([torch.load(path)[:,slices] for path in train_input_file_paths])
        train_targets = torch.cat([torch.load(path) for path in train_target_file_paths])




post = time.time()
print(f"Time to load train: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")



pre = time.time()



if do_fastdataloader == 'no':
    trainloader = DataLoader(allin, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=False)
    total_n_train = len(trainloader.dataset)
else:
    total_n_train = len(train_targets)
    trainloader = FastTensorDataLoader(train_inputs, train_targets, batch_size=bsize, shuffle=True, n_data=total_n_train)
    del train_inputs
    del train_targets
    gc.collect()
    
post = time.time()
print(f"Time to create trainloader: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")


total_len_train = len(trainloader)
print(total_n_train,'\ttraining samples', '\ttotal_len_train', total_len_train)


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


if prev_epochs > 0:
    checkpoint = torch.load(f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_{n_samples}/model_{prev_epochs}_epochs{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_datasets_{n_samples}.pt', map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

print(model)

model.to(device)


if weighting_method == '_ptetaflavloss' or weighting_method == '_altptetaflavloss':
    # loss weighting happens after loss for each sample has been calculated (see later in main train loop), will multiply with sample weights and only afterwards calculate the mean, therefore reduction has to be set to 'none' (performs no dim. red. and gives tensor of length n_batch)
    
    # implementation of focal loss uses the external module as an alternative to CrossEntropyLoss
    if do_FL == 'no':
        criterion = nn.CrossEntropyLoss(reduction='none')
    elif do_FL == 'yes':
        criterion = FocalLoss(alpha, gamma, reduction='none')
else:
    # with this weighting, loss weighting is not necessary anymore (the imbalanced classes are already handled with weights for the custom sampler)
    criterion = nn.CrossEntropyLoss()
    
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)


if prev_epochs > 0:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
else:
    print(f'Learning rate (initial): {lrate}')

def new_learning_rate(ep):
    for g in optimizer.param_groups:
        # not recursive, just iterative calculation based on the initial learning rate stored in the variable lrate and the currect epoch ep
        g['lr'] = lrate/(1+ep/30)                      # decaying learning rate (the larger the number, e.g. 50, the slower the decay, think: after x epochs, the learning rate has been halved)
        #g['lr'] = 0.00001                             # change lr (to a new constant)
        print('lr: ', g['lr'], 'after update')
        
        
#The training algorithm
tic = time.time()
loss_history, val_loss_history = [], []
stale_epochs, min_loss = 0, 10
max_stale_epochs = 100

times = []

with open(f"/home/um106329/aisafety/jet_flavor_MLPhysics/status_logfiles/logfile{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_files_{n_samples}_jets.txt", "a") as log:
    log.write(f"{np.floor((tic-start)/60)} min {np.ceil((tic-start)%60)} s"+' Everything prepared for main training loop.\n')




for e in range(epochs):
    times.append(time.time())
    if prev_epochs+e >= 1:  # this is to keep track of the total number of epochs, if the training is started again multiple times after some epochs that were already done
        new_learning_rate(prev_epochs+e)  # and if it's not the first epoch, decrease the learning rate a tiny bit (see function above) --> takes large steps at the beginning, and small ones close to the end
    running_loss = 0
    model.train()
    for b, (i,j) in enumerate(trainloader):
        if e == 0 and b == 1:
            tb1 = time.time()
            print('first batch done')
            print(f"Time for first batch: {np.floor((tb1-times[0])/60)} min {((tb1-times[0])%60)} s")
            
            with open(f"/home/um106329/aisafety/jet_flavor_MLPhysics/status_logfiles/logfile{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_files_{n_samples}_jets.txt", "a") as log:
                log.write(f"{np.floor((tb1-start)/60)} min {np.ceil((tb1-start)%60)} s"+' First batch done!\n')
        
        if weighting_method == '_ptetaflavloss' or weighting_method == '_altptetaflavloss':
            sample_weights = i[:, -1].to(device, non_blocking=True)  # extract the inputs only from the dataloader returns (remaining: weights)
            i = i[:, :-1].to(device, non_blocking=True)
        else: 
            i = i.to(device, non_blocking=True)
            
        j = j.to(device, non_blocking=True)
        
        
        optimizer.zero_grad()
        
        if epsilon > 0:
            output = model(fgsm_attack(epsilon,i,j,model,criterion,dev=device,filtered_indices=used_variables,restrict_impact=restrict))
        else:
            output = model(i)
            
        loss = criterion(output, j)         
        del i
        del j
        gc.collect()
        
        if weighting_method == '_ptetaflavloss' or weighting_method == '_altptetaflavloss':
            # https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/4
            loss = (loss * sample_weights / sample_weights.sum()).sum()
            loss.mean().backward()
        else:
            loss.backward()
        optimizer.step()
        loss = loss.item()
        running_loss += loss
        del output
        gc.collect()
    else:
        del loss
        gc.collect()
        if e == 0:
            tep1 = time.time()
            print('first training epoch done, now starting first evaluation')
             
            with open(f"/home/um106329/aisafety/jet_flavor_MLPhysics/status_logfiles/logfile{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_files_{n_samples}_jets.txt", "a") as log:
                log.write(f"{np.floor((tep1-start)/60)} min {np.ceil((tep1-start)%60)} s"+' First training epoch done! Start first evaluation.\n')
            
            
            
            ##### LOAD VAL SAMPLES #####

            pre = time.time()
            if do_fastdataloader == 'no':
                allval = ConcatDataset([TensorDataset(torch.load(val_input_file_paths[f])[:,slices], torch.load(val_target_file_paths[f])) for f in range(NUM_DATASETS)])
            else:
                val_inputs = torch.cat([torch.load(path)[:,slices] for path in val_input_file_paths])
                val_targets = torch.cat([torch.load(path) for path in val_target_file_paths])
            


            post = time.time()
            print(f"Time to load val: {np.floor((post-pre)/60)} min {np.ceil((post-pre)%60)} s")
            
            if do_fastdataloader == 'no':
                valloader = DataLoader(allval, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=False)
                total_n_val = len(valloader.dataset)
            else:
                total_n_val = len(val_targets)
                valloader = FastTensorDataLoader(val_inputs, val_targets, batch_size=bsize, shuffle=False, n_data=total_n_val)
                del val_inputs
                del val_targets
                gc.collect()
            postval = time.time()
            print(f"Time to create valloader: {np.floor((postval-post)/60)} min {np.ceil((postval-post)%60)} s")
            
            total_len_val = len(valloader)
            print(total_n_val,'\tvalidation samples', '\ttotal_len_val', total_len_val)
           
            
        with torch.no_grad():
            model.eval()
            if e > 0:
                del vloss
                del val_output
                gc.collect()
            running_val_loss = 0
            for i,j in valloader:
                i = i.to(device, non_blocking=True)
                j = j.to(device, non_blocking=True)
                val_output = model(i)
                vloss = criterion(val_output, j)
                del i
                del j
                gc.collect()
                
                if weighting_method == '_ptetaflavloss' or weighting_method == '_altptetaflavloss':
                    vloss = vloss.mean().item()
                else:
                    vloss = vloss.item()
                running_val_loss += vloss
           
            val_loss_history.append(running_val_loss/total_len_val)
            
            

            if stale_epochs > max_stale_epochs:
                print(f'training stopped by reaching {max_stale_epochs} stale epochs.                                                              ')
                e = e - 1
                break
            if running_val_loss < min_loss:
                min_loss = running_val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                
            print(f"{(e+1)/epochs*100}% done. Epoch: {prev_epochs+e}\tTraining loss: {running_loss/total_len_train}\tValidation loss: {running_val_loss/total_len_val}",end='\n')
            
        loss_history.append(running_loss/total_len_train)
        
        
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": running_loss/total_len_train, "val_loss": running_val_loss/total_len_val}, f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_datasets_{n_samples}.pt')
        # I'm now saving the model both on /hpcwork (fast and mounted for the batch system) and /home (slow, but there are backups)
        torch.save({"epoch": prev_epochs+e, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": running_loss/total_len_train, "val_loss": running_val_loss/total_len_val}, f'/home/um106329/aisafety/jet_flavor_MLPhysics/saved_models/{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_datasets_{n_samples}.pt')
toc = time.time()

print(f"Time elapsed: {np.floor((toc-tic)/60)} min {np.ceil((toc-tic)%60)} s")
print(f"used {NUM_DATASETS} files, {prev_epochs+epochs} epochs, dropout 0.1 4x, learning rate {lrate}")


torch.save(model, f'/hpcwork/um106329/jet_flavor_MLPhysics/saved_models/{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_datasets_{n_samples}_justmodel.pt')

torch.save(model, f'/home/um106329/aisafety/jet_flavor_MLPhysics/saved_models/{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_{n_samples}/model_{prev_epochs+(e + 1)}_epochs{weighting_method}{fl_text}{at_text}{restrict_text}_{NUM_DATASETS}_datasets_{n_samples}_justmodel.pt')

times.append(toc)
for p in range(len(times)-1):
    print(f"Time for epoch {prev_epochs+p}: {np.floor((times[p+1]-times[p])/60)} min {((times[p+1]-times[p])%60)} s")
end = time.time()
print(f"Total time for whole script: {np.floor((end-start)/60)} min {np.ceil((end-start)%60)} s")
