# Gaussianization Flows
This repo contains the implementation for [Gaussianization Flows](https://arxiv.org/abs/2003.01941).

-------------------------------------------------------------------------------------
Iterative Gaussianization is a fixed-point iteration procedure that can transform any continuous random vector into a Gaussian one. 
Based on iterative Gaussianization, we propose a new type of normalizing flow model that enables both efficient 
computation of likelihoods and efficient inversion for sample generation. We demonstrate that these models, 
named Gaussianization flows, are universal approximators for continuous probability distributions under some regularity 
conditions. Because of this guaranteed expressivity, they can capture multimodal target distributions without compromising 
the efficiency of sample generation. Experimentally, we show that Gaussianization flows achieve better or comparable 
performance on several tabular datasets compared to other efficiently invertible flow models 
such as Real NVP, Glow and FFJORD. In particular, Gaussianization flows are easier to initialize, 
demonstrate better robustness with respect to different transformations of the training data, 
and generalize better on small training sets.


## Dependencies

* PyTorch

* seaborn

## Running Experiments
### RBIG Experiments
To run RBIG experiments, simply run
``python rbig.py``
### Tabular Dataset Experiments
To download tabular dataset, follow the instructions [here](https://github.com/gpapamak/maf).
To run the experiments, run 
```
python tabular_experiment.py --multidim_kernel --usehouseholder
```
and specify the dataset and settings by using the flags
```angular2
--total_datapoints  --process_size --dataset --layer --epoch 
--lr --batch_size
```