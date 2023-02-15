# DXtreMM_TSF

Repository for paper, "Deep Extreme Mixture Model for Time series forecasting"

This provides implementation of VD-AE classifier model and forecaster modules

VD-AE classifier is the extension of work titled "Variational Disentanglement for Rare Event Modeling". GPD prior is extended for left extremes. The complete implemetation is given in VD_AE_classifer folder

training the classifier model can be done by executing

python train_VIE_multi_simulationDL_cv.py

Anomaly detection results are saved in same folder

Forecaster modules training and inference implementaion is given in extreme_gpd folder

extreme_gpd_3cls.ipynb invokes training function and infers predicton of unseen data

-----------------------
1. install pytorch (you can follow instructions from [here](https://pytorch.org/get-started/locally/)
2. pip install -r requirements.txt
3. Change parameters according to your data in train_VIE_multi_simulationDL_cv.py file
4. Get classification results and use that for forecasting module

Requirements file can be installed using conda environemnt by following instructions from [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or using docker as given [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
 

