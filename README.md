# Ischemic-Stroke-Lesion-Segmentation with Unet Families
This repository contains the implementation of Unet Families for the task of segmenting 
Ischemic Stroke Lesion.

![sample output](https://github.com/SiqiHuang18/Ischemic-Stroke-Lesion-Segmentation/blob/main/output/att1.png)

- DLM_final_ipynb  contains training , evaluation and experiment code
- layers.py    contains some special layers implemented for networks. It need to be put in the same fold as DLM_final_ipynb
- cross-validation  contains csv for five fold cross-validation
- train_dataloader_final_v3.csv  and  validate_dataloader_final_v3 are csv to be load for five channels
- train_dataloader_final.csv   and  validate_dataloader_final.csv are csv to be load for one channel

- All five-fold validation experiment and five channel experiment are run with same hyperparameters as one channel experiments

## Methods

* Unet

* Attention Unet (2D/3D)

* Multi-resolution Unet (2D/3D)



