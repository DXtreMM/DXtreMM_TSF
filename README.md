# DXtreMM_TSF

Repository for paper, **"Deep Extreme Mixture Model for Time series forecasting"** CIKM ACM conference, 2022 [paper](https://dl.acm.org/doi/10.1145/3511808.3557282)

This model combines two modules:
    1) Variation Disentangled Autoencoder based classifier
    2) GPD based and Normal Forecaster modules

**VD-AE classifier** is the extension of work titled "Variational Disentanglement for Rare Event Modeling". GPD prior is extended for left extremes. 

The complete implemetation of classifier modules is given in VD_AE_classifer folder

Training the classifier model can be done by executing

```
python train_VIE_multi_simulationDL_cv.py
```

Anomaly detection results are saved in same folder

**Forecaster modules** training and inference implementaion is given in extreme_gpd folder

extreme_gpd_3cls.ipynb invokes training function and infers predicton of unseen data

Step by step procedure for working with code is as follows:

-----------------------
```
1. install pytorch (you can follow instructions from [here](https://pytorch.org/get-started/locally/)
2. pip install -r requirements.txt
3. Change parameters according to your data in train_VIE_multi_simulationDL_cv.py file
4. Get classification results and use that for forecasting module
```

Requirements file can be installed using conda environemnt by following instructions from [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or using docker as given [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
 

