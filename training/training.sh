#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --output=output.%J.txt

# with gpu: 

#SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=<your-email>

cd /home/um106329/aisafety/jet_flavor_MLPhysics/training
module unload intelmpi; module switch intel gcc
module load cuda/11.0
module load cudnn
source ~/miniconda3/bin/activate
conda activate my-env
python3 training.py ${FILES} ${PREVEP} ${ADDEP} ${WM} ${NJETS} ${FASTDATALOADER} ${FOCALLOSS} ${GAMMA} ${ALPHA1} ${ALPHA2} ${ALPHA3} ${EPSILON} ${RESTRICT}