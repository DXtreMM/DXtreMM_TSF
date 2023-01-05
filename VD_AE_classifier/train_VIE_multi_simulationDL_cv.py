from __future__ import print_function

import math
import os

import numpy as np
import pandas

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import normal
import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.utils.data
import torchvision

# from data.simulation import simulation_cox_weibull, formatted_data_simu, saveDataCSV
from data.EVT_dataloader import EVTDataset, EVTDataset_dic,ImbalancedDatasetSampler, callback_get_label
from utils.distributions import mixed_loglikeli, loglog_function, sample_mixedGPD, log_sum_exp

from multiclassification.VIEVT_multi_outHz import IAF, Decoder_multiclass, Nu, log_score_marginal, Decoder_VAE
from multiclassification.VIEVT_multi_outHz import testing_VIEVT, pred_avg_risk
from utils.metrics import binary_cross_entropy
from data.simulation_multi_DLN import generate_data, generate_data_semi, saveDataCSV, loadDataCSV, split_t 
from utils.preprocessing import count_rates
from data.simulation_DLN import formatted_data_simu, saveDataCSV, loadDataCSV, formatted_data_simu_noz
# from utils.metrics import binary_cross_entropy,  view_distribution_z_e_hz, view_z_e, view_z_box, view_z_dist, view_distribution, view_z_e
from utils.metrics import cross_entropy, get_predicted_label,  accuracy_per_class
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import TimeSeriesSplit

from pathlib import Path

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

file_path = './data/'
result_path_root = './results/'+"multi/"+"simulationDL"+'/'
Path(result_path_root).mkdir(parents=True, exist_ok=True)

# lambda_=[1.0, 5e-3, 1e-5]
lambda_=[1.0, 5e-3, 1e-6]
# lambda_=[1.0, 1e-4, 1e-6]
hidden_layer_MNN = [32,32]

seed=123


##=================Things to change ===========

data_name='B_SPI_testing'
lag=9
num_prev = 5
num_next = 5
df = pd.read_csv('B_SPI_testing.csv')
time_series = df['SPI1'].to_numpy()

##==============================================

# data_name = 'seattle'
# data_name = 'pr_1901_2020_IND'
# data_name = 'Metro_Interstate_Traffic_Volume'
# data_name = 'co2_mm_mlo'
# data_name = 'ETTh1'
# data_name='nyc-yellow-taxi'
#data_name='nasdaq100_padding'
# er005
# cut_bound=0.10
# df, continuous_variables = generate_data(file_path, result_path_root, cut_bounds, seed)
# print(count_rates(df['e']))

# lag=47
#lag=9
#num_prev = 5
#num_next = 5
# df = pd.read_csv('/home/abilasha/Downloads/extreme_pred/upload/Data/Metro_Interstate_Traffic_Volume/Metro_Interstate_Traffic_Volume.csv')
# df = pd.read_csv('/home/abilasha/Downloads/extreme_pred/upload/Data/co2_mm_mlo.csv')
#df = pd.read_csv('/home/abilasha/Downloads/extreme_pred/upload/Data/nasdaq100/nasdaq100/small/nasdaq100_padding.csv')
# df = pd.read_csv('/home/abilasha/Downloads/extreme_pred/upload/Data/ETDataset/ETT-small/ETTh1.csv')
# df = pd.read_csv('/home/abilasha/Downloads/extreme_pred/upload/Data/rainfall/seattle.csv')
# df = pd.read_csv('/home/abilasha/Downloads/extreme_pred/upload/Data/rainfall/pr_1901_2020_IND.csv')
# df = pd.read_csv('/home/abilasha/Downloads/extreme_pred/upload/Data/nyc_yellow_taxi/nyc-yellow-taxi.csv')

# time_series = df['PRCP'].to_numpy()
# time_series = df['Rainfall'].to_numpy()
# time_series = df['traffic_volume'].to_numpy()
# time_series = df['de-seasonalized'].to_numpy()
#time_series = df['AAPL'].to_numpy()
# time_series = df['OT'].to_numpy()
# time_series = df['count'].to_numpy()

n=time_series.shape[0]
for i in np.where(np.isnan(time_series))[0]:
    values = time_series[max(i - num_prev, 0): min(i + num_next + 1, n)]
    values = values[np.logical_not(np.isnan(values))]
    
    time_series[i] = values.mean()

val_diff = time_series - shift(time_series, 1, cval=time_series[0])
time_series=val_diff[~np.isnan(val_diff)]


extreme_rthreshold = np.quantile(time_series,.98)
extreme_lthreshold = np.quantile(time_series,.02)

n=time_series.shape[0]

train_frac = 0.80
val_frac = 0.10

num_train = int(n * train_frac)
num_val = int(n * val_frac)
num_test = n - num_train - num_val

print(f'Num Train: {num_train}, Num Val: {num_val}, Num Test: {num_test}')

train_test_ranges = []
tscv = TimeSeriesSplit(n_splits=3, test_size=num_test)
for train_index, test_index in tscv.split(time_series):
    train_test_ranges.append(((train_index[0], train_index[-1]), (test_index[0], test_index[-1])))

class_acc_avg_train=[]
class_acc_avg_test=[]
prec_avg_train=[]
prec_avg_test=[]

run_count=0

for (train_low, train_high), (test_low, test_high) in train_test_ranges:
    print(f'Train: {train_low}: {train_high}, Test: {test_low}: {test_high}')

    inputData = []
    outputData = []
    for i in range(train_low+lag, train_high):
        inputData.append(time_series[i - lag: i])
        outputData.append([1.0 if (time_series[i] > extreme_rthreshold) else 2.0 if (time_series[i] < extreme_lthreshold) else 0.0])

    df_train={"x": np.array(inputData), "e": np.array(outputData)}

    inputData = []
    outputData = []
    for i in range(test_low+lag, test_high):
        inputData.append(time_series[i - lag: i])
        outputData.append([1.0 if (time_series[i] > extreme_rthreshold) else 2.0 if (time_series[i] < extreme_lthreshold) else 0.0])

    df_test={"x": np.array(inputData), "e": np.array(outputData)}


    inputData = []
    outputData = []
    for i in range(train_low+lag, test_high):
        inputData.append(time_series[i - lag: i])
        outputData.append([1.0 if (time_series[i] > extreme_rthreshold) else 2.0 if (time_series[i] < extreme_lthreshold) else 0.0])

    df_combined={"x": np.array(inputData), "e": np.array(outputData)}

    np.random.seed(seed)
    n_samples_train = df_train['x'].shape[0]
    n_samples_test = df_test['x'].shape[0]
    n_samples = df_combined['x'].shape[0]
    # print(n_samples)

    train_idx = np.arange(0,int(4*n_samples_train/5))
    valid_idx = np.arange(int(4*n_samples_train/5),n_samples_train)
    test_idx = np.arange(0,n_samples_test)
    test_whole_idx = np.arange(0,n_samples)

    train = formatted_data_simu_noz(df_train['x'],  df_train['e'], train_idx)
    test = formatted_data_simu_noz(df_test['x'], df_test['e'], test_idx)
    valid = formatted_data_simu_noz(df_train['x'], df_train['e'], valid_idx)
    test_whole = formatted_data_simu_noz(df_combined['x'], df_combined['e'], test_whole_idx)

    ncov = lag
    continuous_variables=np.arange(ncov)

    # event_rate = np.min(np.mean(train['e']))
    # ids=np.unique(train['e']) #array of unique ids
    all_rates = count_rates(train['e'])
    n_cat = len(all_rates)
    event_rate=all_rates
    # event_rate=np.min(all_rates)
    rare_class = np.argmin(all_rates)
    print(event_rate)
    # all_rates=np.max(all_rates)/all_rates
    all_rates=1/(all_rates*10)
    print(all_rates)


    # saveDataCSV(train, data_name+'_simulationDL_train', result_path_root)
    # saveDataCSV(valid, data_name+'_simulationDL_valid', result_path_root)
    # saveDataCSV(test, data_name+'_simulationDL_test', result_path_root)
    # np.save(result_path_root+'continuous_variables', continuous_variables)

    # train = loadDataCSV(data_name+'_simulationDL_train', result_path_root)
    # valid = loadDataCSV(data_name+'_simulationDL_valid', result_path_root)
    # test = loadDataCSV(data_name+'_simulationDL_test', result_path_root)
    # continuous_variables = np.load(result_path_root+'continuous_variables'+'.npy')


    result_path = result_path_root +data_name
    Path(result_path).mkdir(parents=True, exist_ok=True)
    model_path = result_path+"/saved_models"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    plot_path = result_path+"/plots"
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    ########## Hyper-parameters##############
    model_name = 'VIE'
    # z_dim = 24
    z_dim=4      ##4
    hidden_layers=[32,32]
    # ncov = train['x'].shape[1]
    # event_rate = train['e'].mean()
    eps_dim = int(ncov)
    input_size = ncov
    # input_size = ncov+eps_dim
    unroll_steps = 5
    nu_lambda=1.0
    epochs = 100
    # epochs=50
    batch_size = 200
    # batch_size=100
    # batch_size=64


    flow_path = result_path+"/saved_models/"+model_name+'_flow_'+str(run_count)+".pt"

    decoder_path = result_path+"/saved_models/"+model_name+'_decoder_'+str(run_count)+".pt"
    nu_path = result_path+"/saved_models/"+model_name+'_nu_'+str(run_count)+".pt"

    # device = torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(1)

    training = True

    unroll_test = True

    u_bound = np.max([0.98, 1-event_rate[1]])
    l_bound=np.min([0.02, event_rate[2]])
    # u_bound=1-event_rate[1]
    # l_bound=event_rate[2]
    # u_bound = np.max([0.98, 1-all_rates[1]])
    # l_bound=np.min([0.02, all_rates[2]])
    lower_bound = -5.0
    N = 10

    IAF_flow = IAF(input_size, z_dim=z_dim, h_dim=z_dim, hidden_layers=hidden_layers, nstep=5, device=device)
    decoder = Decoder_multiclass(z_dim=z_dim, n_category=n_cat,hidden_layer_MNN=hidden_layer_MNN)
    # decoder = Decoder_VAE(z_dim=z_dim, hidden_layers=[32,32])
    nu = Nu(z_dim=z_dim, ncov=ncov, hidden_layers=[32,32], marginal=True)

    decoder.to(device)
    IAF_flow.to(device)
    nu.to(device)

    # define optimizer
    opt_flow = optim.Adam(IAF_flow.parameters(), lr=1e-4)
    opt_dec = optim.Adam(decoder.parameters(), lr=1e-4)
    opt_nu = optim.RMSprop( nu.parameters(), lr = 1e-3)

    aggressive_flag = True
    aggressive_nu = True


    # consider normaliztion of inputs
    norm_mean = np.mean(train['x'][:,continuous_variables],axis=0)
    norm_std = np.std(train['x'][:,continuous_variables],axis=0)

    EVT_train = EVTDataset_dic(train,transform=True,norm_mean=norm_mean, norm_std=norm_std, continuous_variables=continuous_variables)
    EVT_valid = EVTDataset_dic(valid,transform=True,norm_mean=norm_mean, norm_std=norm_std, continuous_variables=continuous_variables)

    
    train_loader = DataLoader(EVT_train, batch_size=batch_size)
    # train_loader = DataLoader(EVT_train, batch_size=batch_size*10, sampler=ImbalancedDatasetSampler(valid, callback_get_label=callback_get_label))
    # validation on the original scale
    valid_loader = DataLoader(EVT_valid, batch_size=10000, shuffle=True)

    print("Dataloading done")

    del train
    ## define aggressive training
    def agrressive_step():
        opt_flow.zero_grad()    
        opt_dec.zero_grad()
        nanFlag =0
        # best_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())
        best_z, likelihood_qzx = IAF_flow(batched_x.float())
        try:
            assert (best_z != best_z).any()== False
        except AssertionError:
            nanFlag =1
            return
        pred_risk_cur = decoder(best_z, N, lower_bound).float()
        # pred_risk_cur = decoder(best_z).float()                                             ##VAE
    #     BCE_loss = binary_cross_entropy(pred_risk_cur, \
    #                                     batched_e.detach().float(), sample_weight=batch_weight.float())
        # print(pred_risk_cur.shape)
        # print(batched_e.shape)
        CE_loss = cross_entropy(pred_risk_cur, batched_e.squeeze().long(), sample_weight=batch_weight.float())

        z_nu, pz_nu, nanFlag_ = log_score_marginal(nu=nu, z=best_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                xi_=IAF_flow.xi_, sigma_u=IAF_flow.sigma_u, sigma_l=IAF_flow.sigma_l,\
                                p_ = u_bound, p_l=l_bound, eps=1e-3, nu_lambda=nu_lambda,device=device, train_nu=False)
        
        # calculate KL(q(z|x)||p(z))
        
        likelihood_pz = mixed_loglikeli(best_z, IAF_flow.mu0, IAF_flow.logvar0, IAF_flow.xi_, IAF_flow.sigma_u, IAF_flow.sigma_l, u_bound, l_bound)
        assert (likelihood_pz != likelihood_pz).any()== False 
        KL_cond = likelihood_qzx.sum() - likelihood_pz.sum()
        loss = lambda_[0]*CE_loss + lambda_[1]*(z_nu - pz_nu) + lambda_[2]*KL_cond
        loss.backward()
        torch.nn.utils.clip_grad_norm_(IAF_flow.parameters(), 1e-4)
        opt_flow.step()
        
        return loss.item()

    print("start training")
    # training process

    if __name__ == "__main__":
        # if training:
            best_valid_loss = np.inf
            best_valid_recon_loss = np.inf
            best_valid_rare_acc = 0
            best_valid_F1 = 0
            best_epoch = 0
            last_shrink = 0
            nanFlag = 0
            
            # save training process
            
            train_z_nu = []
            train_pz_nu = []
            train_KL = []
            train_BCE = []
            last_shrink = 0

            for epoch in range(1, epochs + 1):
                if nanFlag  > 0:
                    break
                    
        #         train(epoch)
        #         test(epoch)
                train_loss = 0
                valid_loss = 0
                valid_recon_loss = 0
                valid_pos_loss = 0
                pre_mi = 0
                
                improved_str = " "

                # detect errors
    #             with torch.autograd.detect_anomaly():
                for batch_idx, batched_sample in enumerate(train_loader):
    #                 print(batch_idx)
                    if nanFlag  > 0:
                        break
                    IAF_flow.train()
                    decoder.train()
                    nu.train()

                    batched_x =  batched_sample['x']
                    batched_x = batched_x.to(device).view(-1, ncov)
                    batched_e =  batched_sample['e'].to(device)
                    batch_weight = torch.Tensor(all_rates).float().to(device)
                    # batch_weight=torch.Tensor([1,30,27]).float().to(device)
                    
                    # add noise
                    eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)                
                    best_z, likelihood_qzx = IAF_flow(batched_x.float())
                    # best_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())
                    # print(best_z)
                    try:
                        assert (best_z != best_z).any()== False
                        
                    except AssertionError:
                        nanFlag = nanFlag+1
                        break
                    # aim to update nu based on conditional q
                    # update multiple times of the critic

                    if aggressive_nu:
                        if epoch > 5:
                            aggressive_nu = False
                            print("STOP multiple learning of nu")
                        for iter_ in range(unroll_steps):
                        ## conditional posterior
                            # aim to update nu based on marginal q
                            z_nu, pz_nu, loss_nu, nanFlag_ = log_score_marginal(nu=nu, z=best_z, \
                                                                               mu=IAF_flow.mu0, logvar=IAF_flow.logvar0,\
                                                                               xi_=IAF_flow.xi_, sigma_u=IAF_flow.sigma_u, sigma_l=IAF_flow.sigma_l,\
                                                                               p_ = u_bound, p_l=l_bound, eps=1e-3, nu_lambda=nu_lambda,\
                                                                               device=device,train_nu=True, opt_nu=opt_nu)           

                            nanFlag = nanFlag + nanFlag_
                            if ((1*torch.isnan(best_z)).sum() + (1*torch.isnan(pz_nu)).sum()  + (1*torch.isnan(z_nu)).sum()).item()>0:
                                print("NaN occured at critic training")
        #                         print(z_init)
                                print(IAF_flow.xi_, IAF_flow.sigma_u, IAF_flow.sigma_l, IAF_flow.mu0, IAF_flow.logvar0)
                                nanFlag = nanFlag + 1
                                break 
                        try:
                            assert (nanFlag>0)== False
                        
                        except AssertionError:
                            nanFlag = nanFlag+1
                            break
                    else:
                        z_nu, pz_nu, loss_nu, nanFlag_ = log_score_marginal(nu=nu, z=best_z,\
                                                                           mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                                                           xi_=IAF_flow.xi_, sigma_u=IAF_flow.sigma_u, sigma_l=IAF_flow.sigma_l,\
                                                                           p_ = u_bound, p_l=l_bound, eps=1e-3, nu_lambda=nu_lambda,\
                                                                           device=device, train_nu=True, opt_nu=opt_nu)                                
                        nanFlag = nanFlag + nanFlag_
                    # update encoder and decoder's parameters
                    try:
                        assert (nanFlag>0)== False

                    except AssertionError:
                        nanFlag = nanFlag+1
                        break              
                    if aggressive_flag:    
                        sub_iter = 0
                        while sub_iter  < 10:

                            sub_loss = agrressive_step()
        #                     print(sub_iter,sub_loss)
                            sub_iter += 1
                        
                    
                    opt_dec.zero_grad()
                    opt_flow.zero_grad()
                    
                    CE_loss = cross_entropy(decoder(best_z, N, lower_bound).float(), batched_e.squeeze().long(), sample_weight=batch_weight.float())
                    # CE_loss = cross_entropy(decoder(best_z).float(), batched_e.squeeze().long(), sample_weight=batch_weight.float())    ##VAE

                    z_nu, pz_nu, nanFlag_ = log_score_marginal(nu=nu, z=best_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                            xi_=IAF_flow.xi_, sigma_u=IAF_flow.sigma_u, sigma_l=IAF_flow.sigma_l,\
                                            p_ = u_bound, p_l=l_bound, eps=1e-3, nu_lambda=nu_lambda,device=device, train_nu=False)
                    nanFlag = nanFlag + nanFlag_
                    try:
                        assert (nanFlag>0)== False

                    except AssertionError:
                        nanFlag = nanFlag+1
                        break                
                    likelihood_pz = mixed_loglikeli(best_z, IAF_flow.mu0, IAF_flow.logvar0, IAF_flow.xi_, IAF_flow.sigma_u, IAF_flow.sigma_l, u_bound, l_bound)
                    KL_cond = likelihood_qzx.sum() - likelihood_pz.sum()
    #                 print(likelihood_qzx, likelihood_pz.sum())
                    # print(loss.item())
                    loss = lambda_[0]*CE_loss + lambda_[1]*(z_nu - pz_nu) + lambda_[2]*KL_cond
                    loss.backward()
                    
        
                    train_z_nu.append(z_nu.item())
                    train_pz_nu.append(pz_nu.item())
                    train_BCE.append(CE_loss.item())
                    train_KL.append(KL_cond.item())
                    
                    
                    train_loss += loss.item()
                    if not aggressive_flag:
                        torch.nn.utils.clip_grad_norm_(IAF_flow.parameters(), 1e-4)
                        opt_flow.step()

                    opt_dec.step()
                    if nanFlag > 0:
                            break
                    
                                
                
                print('====> Epoch: {} Average loss: {:.4f}'.format(
                              epoch, train_loss))

                    
                try:
                    assert (nanFlag>0)== False

                except AssertionError:
                    nanFlag = nanFlag+1
                    break
                # check performance on validation dataset
    #             with torch.no_grad():
                if nanFlag == 0:
                    IAF_flow.eval()
                    decoder.eval()
                    nu.eval()
                    for i, batched_sample in enumerate(valid_loader):
                        batched_x =  batched_sample['x']
                        batched_x = batched_x.to(device).view(-1, ncov)
                        batched_e =  batched_sample['e'].to(device)
                        # add noise
                        eps_ = (torch.Tensor( batched_x.shape[0], eps_dim).normal_()).to(device)
                        batch_z, likelihood_qzx = IAF_flow(batched_x.float())
                        # batch_z, likelihood_qzx = IAF_flow(batched_x.float(), eps_.float())

                        if aggressive_flag:
                            if epoch > 1:
                                aggressive_flag = False
                            
    #                         cur_mi = likelihood_qzx.sum() - (log_sum_exp(likelihood_qzx)).sum()
    #                         if cur_mi - pre_mi < 0:
    #                             aggressive_flag = False
    #                             print("STOP aggressive learning")
    #                         cur_mi = pre_mi
                            
                        
    #                     pred_risk_batch = decoder(batch_z, N, lower_bound)
                        pred_risk_batch, likelihood_qzx= pred_avg_risk(batched_x, eps_dim, IAF_flow, decoder, device, n_avg=1)
                        pred_label = get_predicted_label(pred_risk_batch.detach().cpu())
        
                        if (pred_risk_batch !=pred_risk_batch).any()==True:
                            break
                    
                        # valid_recon_, valid_acc_ = cross_entropy(pred_risk_batch, batched_e.squeeze().long(),\
                        #                              class_acc=True)
                        valid_recon_, valid_acc_, cm= cross_entropy(pred_risk_batch, batched_e.squeeze().long(),\
                                                     sample_weight=batch_weight.float().to(device), class_acc=True)

                        print(np.unique(pred_label))
           
                        # based on marginal q
                        z_nu, pz_nu,nanFlag_ = log_score_marginal(nu=nu, z=batch_z, mu=IAF_flow.mu0, logvar=IAF_flow.logvar0, \
                                                xi_=IAF_flow.xi_, sigma_u=IAF_flow.sigma_u, sigma_l=IAF_flow.sigma_l,\
                                                p_ = u_bound, p_l=l_bound, eps=1e-3, device=device, train_nu=False)
                        nanFlag = nanFlag + nanFlag_
                        # based on conditional q
                        likelihood_pz = mixed_loglikeli(batch_z, IAF_flow.mu0, IAF_flow.logvar0, IAF_flow.xi_, IAF_flow.sigma_u, IAF_flow.sigma_l, u_bound, l_bound)
                        KL_cond = likelihood_qzx.sum()  - likelihood_pz.sum()    
                
                        valid_loss_ = valid_recon_ + z_nu - pz_nu + KL_cond
                        # calculating F1 score                    
                        valid_F1 = sklearn.metrics.f1_score(pred_label,\
                                            batched_sample['e'].cpu().squeeze().numpy(),average='micro')
                        # valid_rare_acc = valid_acc_[rare_class].item()
                        valid_rare_acc = valid_acc_[1].item()
                        valid_rare_acc += valid_acc_[2].item()

    #                     # calculating AUC
    #                     pred_risk = pred_risk_batch.cpu().detach().squeeze().numpy()
    #                     nonnan_idx = np.where(np.isnan(pred_risk)==False)[0]
    #                     pred_risk = pred_risk[nonnan_idx]
    #                     valid_auc_ = sklearn.metrics.roc_auc_score(batched_sample['e'][nonnan_idx,:].cpu().squeeze().numpy(),\
    #                                                                pred_risk).item()
    #                     # calculating F1 score                    
    #                     valid_F1 = F1_score(batched_sample['e'].cpu().squeeze().numpy(),\
    #                                         pred_risk_batch.cpu().detach().squeeze().numpy(), beta=1.0)
                        
                        valid_loss = valid_loss + valid_loss_.item()
                        valid_recon_loss = valid_recon_loss + valid_recon_.item()
    #                     valid_pos_loss = valid_pos_loss + pos_recon_.item()
                        break


                    if np.isnan(valid_recon_loss) == False:
                        save_model = 0
                        if (valid_recon_loss < best_valid_recon_loss) or (valid_rare_acc > best_valid_rare_acc) or (valid_F1 > best_valid_F1):
                            if (valid_recon_loss < best_valid_recon_loss):
        #                         best_valid_recon_loss = valid_recon_loss
            #                     torch.save(model.state_dict(), model_path)
                                save_model += 1
                            if (valid_rare_acc > best_valid_rare_acc):
        #                         best_valid_pos_loss = valid_pos_loss
                                save_model += 1
                            if (valid_F1 > best_valid_F1):
        #                         best_valid_auc = valid_auc_
                                save_model += 1


                            # save current model

                        if save_model > 1:
                            # Save current metrics as standard
                            best_valid_pos_loss = valid_pos_loss
                            best_valid_valid_rare_acc = valid_rare_acc
                            best_valid_recon_loss = valid_recon_loss
                            best_valid_F1 = valid_F1
                            best_epoch = epoch
                            torch.save(IAF_flow.state_dict(), flow_path)
                            torch.save(decoder.state_dict(), decoder_path)
                            torch.save(nu.state_dict(), nu_path)
                            improved_str = "*"
        #                     prior_z = sample_mixedGPD(8000,  mu=IAF_flow.mu0, logvar=IAF_flow.logvar0,\
        #                                               xi_=IAF_flow.xi_, sigma_=IAF_flow.sigma_,\
        #                                               p_ = u_bound, lower_bound = -5.0, upper_bound = 50, device=device)
                            # view_distribution(batch_z, prior_z, model_name, plot_path)

                    if (epoch - best_epoch >=10) and (epoch - last_shrink >=10):
                        lambda_[1] = lambda_[1] * 5e-1
                        lambda_[2] = lambda_[2] * 5e-1
                        last_shrink = epoch


                    print('====> Valid CE loss: {:.4f}\t  Class Accuracy: {} KL Loss: {:.4f}  z_nupz_nu: {:.4f} \tImproved: {}'.format(valid_recon_loss, valid_acc_, KL_cond, (z_nu - pz_nu), improved_str))

                    if epoch - best_epoch >=10: ##this was 20
                        print('Model stopped due to early stopping')
                        break
               
            all_idx = np.arange(0,n_samples_train)
            train = formatted_data_simu_noz(df_train['x'],  df_train['e'], all_idx)

            # report results in testing        
            pred_label_risk, batch_z, class_acc_, test_F1, prec_train = testing_VIEVT("train", run_count, train, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, transform = True, norm_mean=norm_mean, norm_std=norm_std,  N=N, continuous_variables=continuous_variables, device=device, saveResults=True)
            # pred_label_risk, batch_z, Hz, class_acc_, test_F1 = testing_VIEVT(test, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, transform = True, norm_mean=norm_mean, norm_std=norm_std,  N=N, continuous_variables=continuous_variables, device=device, saveResults=True)
            class_acc_avg_train.append(class_acc_)
            prec_avg_train.append(prec_train)

            pred_label_risk, batch_z, class_acc_, test_F1, prec_test = testing_VIEVT("test", run_count, test, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, transform = True, norm_mean=norm_mean, norm_std=norm_std,  N=N, continuous_variables=continuous_variables, device=device, saveResults=True)
            # pred_label_risk, batch_z, Hz, class_acc_, test_F1 = testing_VIEVT(test, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, transform = True, norm_mean=norm_mean, norm_std=norm_std,  N=N, continuous_variables=continuous_variables, device=device, saveResults=True)
            class_acc_avg_test.append(class_acc_)
            prec_avg_test.append(prec_test)


            pred_label_risk, batch_z, class_acc_, test_F1, prec_train = testing_VIEVT("test_whole", run_count, test_whole, IAF_flow, flow_path, decoder, decoder_path, nu, nu_path, model_name, result_path, eps_dim, transform = True, norm_mean=norm_mean, norm_std=norm_std,  N=N, continuous_variables=continuous_variables, device=device, saveResults=True)
            
            run_count+=1

print('average class accuracy_train:{}, \t std class accuracy_train:{}, \n average class accuracy_test:{}, \t std class accuracy_test:{}' .format(np.mean(class_acc_avg_train, axis=0), np.std(class_acc_avg_train, axis=0), np.mean(class_acc_avg_test,axis=0), np.std(class_acc_avg_test, axis=0) ))
print('average class prec_train:{}, \t std class prec_train:{}, \n average class prec_test:{}, \t std class prec_test:{}' .format(np.mean(prec_avg_train, axis=0), np.std(prec_avg_train, axis=0), np.mean(prec_avg_test,axis=0), np.std(prec_avg_test, axis=0) ))
