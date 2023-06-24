# Adversarial-Training-for-Jet-Tagging
Code for:
> <b><a href="https://arxiv.org/abs/2203.13890" target="_blank">Improving robustness of jet tagging algorithms with adversarial training</a></b>  
> A. Stein et al.
> Comput.Softw.Big Sci., 2022.

<i>Jet Flavor dataset</i>

Obtained from http://mlphysics.ics.uci.edu/ and originally created for 
> <b><a href="https://arxiv.org/abs/1607.08633" target="_blank">Jet Flavor Classification in High-Energy Physics with Deep Neural Networks</a></b>  
> D. Guest et al.
> Physical Review D, 2016.

## Get and prepare dataset
### Download
Login to a copy18-node of the HPC with high bandwith (will download 2.2GB)
```
wget http://mlphysics.ics.uci.edu/data/hb_jet_flavor_2016/dataset.json.gz
mkdir -p /hpcwork/<your-account>/jet_flavor_MLPhysics/dataset
mv dataset.json.gz /hpcwork/<your-account>/jet_flavor_MLPhysics/dataset
```
### Extracting the data via awkward arrays
Actually, it turns out that reading the file is not straightforward, at some point, the data has to be unzipped or extracted. The file might have the simple ending ".json", but it's rather various JSON-like entries distributed over several lines of the entire .json file. Consult the notebook `preparations/read_dataset.ipynb` for further details and potential alternatives to use the dataset. Finally, I ended up using awkward arrays with which the next steps become a bit easier.
### A first look at the data
Some initial investigations before proceeding to the actual framework will be conducted inside `preparations/explore_dataset.ipynb`.
### Calculate defaults
To use custom default values that fit well to the bulk distribution, preliminary studies are done inside `preparations/defaults.ipynb`. It's also the first notebook that makes use of `helpers/variables.py`.
### Clean samples
In order to not store too many versions of the same data, cleaning the samples will not be done as a separate step, but comes later when doing the preprocessing (scaling). There, also the final shape of the arrays will be flattened, the result should be a set of usable PyTorch tensors. During the cleaning, I would not cut on any variables, but would only modify certain unphysical values and place them at special default bins - i.e. the fractions of jets of a certain flavor, in certain pt and eta bins do not change by the next step of cleaning (and preprocessing) the data.
### Calculate sample weights
Sample weights are calculated in `preparations/reweighting.ipynb`.
### Preprocessing
Calculate scalers (from trainset only, and ignore defaults), apply scalers (do _not_ ignore defaults when applying the scaler, alternative: set to zero), train/val/test splitting & shuffling, build sample weights and bins. See `preparations/clean_preprocess.ipynb` for a first working example of the entire preprocessing chain. Also, `evaluate/tools.py` can be used later to facilitate communication between training or evaluation scripts with the preprocessing step.
## Run framework (training, evaluation)
### Training
All relevant scripts are placed inside `training`, e.g. standalone training on current node is done with `training.py`, and for submission to the batch system, there is `training.sh` and `submit_training.py`. Can use nominal or adversarial training.
### Evaluation
ROC curves: `evaluate/eval_roc_new.py`. Training history (loss): `evaluate/plot_loss.py`. Tagger outputs and discriminator shapes: `evaluate/eval_discriminator_shapes.py`. Plotting of input variables `evaluate/eval_inputs.py`.
