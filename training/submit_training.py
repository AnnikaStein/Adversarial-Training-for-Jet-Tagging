#!/usr/bin/python3
## file: submit_training.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess

import numpy as np

import argparse

# example: python submit_training.py -f 230 -p 0 -a 10 -w _ptetaflavloss -j -1 -l yes -fl no -g 0 -al1 equal -al2 equal -al3 equal -eps -1 -r -1

parser = argparse.ArgumentParser(description="Setup for training")
parser.add_argument('-f',"--files", type=int, help="Number of files for training", default=230)
parser.add_argument('-p',"--prevep", type=int, help="Number of previously trained epochs", default=0)
parser.add_argument('-a',"--addep", type=int, help="Number of additional epochs for this training", default=30)
parser.add_argument('-w',"--wm", help="Weighting method", default="_ptetaflavloss")
parser.add_argument('-j',"--jets", type=int, help="Number of jets, if one does not want to use all jets for training, if all jets shall be used, type -1", default=-1)
parser.add_argument('-l',"--dofastdl", help="Use fast DataLoader", default='yes')
parser.add_argument('-fl',"--dofl", help="Use Focal Loss", default='yes')
parser.add_argument('-g',"--gamma", type=int, help="Gamma (exponent for focal loss)", default=2)
parser.add_argument('-al1',"--alpha1", help="Alpha (prefactor for focal loss) for category 1 or type 'equal' for no additional weights.", default='equal')
parser.add_argument('-al2',"--alpha2", help="Alpha (prefactor for focal loss) for category 2 or type 'equal' for no additional weights.", default='equal')
parser.add_argument('-al3',"--alpha3", help="Alpha (prefactor for focal loss) for category 3 or type 'equal' for no additional weights.", default='equal')
parser.add_argument('-eps',"--epsilon", type=float, help="Do Adversarial Training with epsilon > 0, or put -1 to do basic training only.", default=-1.0)
parser.add_argument('-r',"--restrict", help="Restrict impact of the attack ? -1 for no, some positive value for yes", default=-1)
args = parser.parse_args()

NUM_DATASETS = args.files
prev_epochs = args.prevep
epochs = args.addep
weighting_method = args.wm
n_samples = args.jets
do_fastdataloader = args.dofastdl
do_FL = args.dofl
gamma = args.gamma
alphaparse1 = args.alpha1
alphaparse2 = args.alpha2
alphaparse3 = args.alpha3
epsilon = args.epsilon
restrict = args.restrict
    
    
home = os.path.expanduser('~')
logPath = home + "/aisafety/jet_flavor_MLPhysics/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/jet_flavor_MLPhysics/training/"
print('Shell script is located at:\t',shPath)

time = 14
mem = 14
factor_FILES = NUM_DATASETS / 230.0

factor_EPOCHS = epochs / 35.0
if epsilon > 0:
    factor_EPOCHS *= 1.55
    if float(restrict) > 0:
        time *= 1.5
        mem *= 1.1


if NUM_DATASETS == 230:
    time = int(np.rint(time * factor_EPOCHS + 2))

else:
    time = int(np.rint(time * factor_FILES * factor_EPOCHS) + 2)

    mem = min(int(np.rint(mem * factor_FILES) + 10),50)

    
submit_command = ("sbatch "
        "--time=00:{5}:00 "
        "--mem-per-cpu={4}G "
        "--job-name=tr_{0}_{1}_{2}{3}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15} "
        "--export=FILES={0},PREVEP={1},ADDEP={2},WM={3},NJETS={7},FASTDATALOADER={8},FOCALLOSS={9},GAMMA={10},ALPHA1={11},ALPHA2={12},ALPHA3={13},EPSILON={14},RESTRICT={15} {6}training.sh").format(NUM_DATASETS, prev_epochs, epochs, weighting_method, mem, time, shPath, n_samples, do_fastdataloader, do_FL, gamma, alphaparse1, alphaparse2, alphaparse3, epsilon, restrict)

print(submit_command)
userinput = input("Submit job? (y/n) ").lower() == 'y'
if userinput:
    exit_status = subprocess.call(submit_command, shell=True)
    if exit_status==1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))
    print("Done submitting jobs!")
