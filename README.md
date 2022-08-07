# DXtreMM_TSF

Repository for paper, Deep Extreme Mixture Model for Time series forecasting

This provides implementation of VD-AE classifier model and forecaster modules

VD-AE classifier is the extension of work titled "Variational Disentanglement for Rare Event Modeling". GPD prior is extended for left extremes. The complete implemetation is given in VD_AE_classifer folder

training the classifier model can be done by executing

python train_VIE_multi_simulationDL_cv.py

Anomaly detection results are saved in same folder

Forecaster modules training and inference implementaion is given in extreme_gpd folder

extreme_gpd_3cls.ipynb invokes training function and infers predicton of unseen data
