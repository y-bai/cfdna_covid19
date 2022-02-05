'''


author: Yong Bai on 2021/10/14

'''

import os
import numpy as np
import pandas as pd
import pickle
import json
from functools import partial 
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL

from scipy import special

import lightgbm as lgb

from scipy.misc import derivative

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss,f1_score
from sklearn.metrics import precision_recall_curve,average_precision_score
import shap

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def weighted_log_loss(pred, dtrain, w_alpha):
        # assign the value of imbalanced alpha
        imbalance_alpha = w_alpha
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        grad = -(imbalance_alpha ** label) * (label - sigmoid_pred)
        hess = (imbalance_alpha ** label) * sigmoid_pred * (1.0 - sigmoid_pred)

        return grad, hess
    
def weighted_ll(pred, dtrain, w_alpha):
        # assign the value of imbalanced alpha
        imbalance_alpha = w_alpha
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        
        sigmoid_pred = np.clip(sigmoid_pred, 1e-15, 1 - 1e-15)
        
        weighted_ll = -(imbalance_alpha * label * np.log(sigmoid_pred)+(1-label)*np.log(1-sigmoid_pred))
        return 'weighted_log_loss', np.mean(weighted_ll), False

def sigmoid(x): return 1./(1. +  np.exp(-x))

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    
    y_hat = sigmoid(y_hat)
    y_hat = np.where(y_hat < 0.5, 0, 1) # scikits f1 doesn't like probabilities
    return 'f1_loss', f1_score(y_true, y_hat), True


def wbc_cv_train(params, trn_data_df, n_fold=5, final_cv_model=False):
    '''
    '''
    params['metric']='auc'
    params['verbose']=-1
    
    cls_param = params.copy()
    
    wbc_alpha = params['wbc_alpha']
    num_iterations = params['num_iterations']
    
    del cls_param['wbc_alpha']
    del cls_param['num_iterations']
    
    wbc_ll = lambda x,y: weighted_log_loss(x, y,wbc_alpha)
    
    x_train = trn_data_df.iloc[:,2:].copy().reset_index(drop=True)
    y_train = trn_data_df[['new_triage']].copy().reset_index(drop=True)
    
    kf = StratifiedKFold(n_splits=n_fold,random_state=0, shuffle=True) 
    metric_score=[]
    for tr_indx,val_index in kf.split(x_train, y_train):
    
        X_tr_data = x_train.iloc[tr_indx,:]
        X_val_data = x_train.iloc[val_index,:]
        
        y_tr_data = y_train.loc[tr_indx,'new_triage'].values
        y_val_data = y_train.loc[val_index,'new_triage'].values
        
        i_trn_dt = lgb.Dataset(X_tr_data, label=y_tr_data)
        i_val_dt = lgb.Dataset(X_val_data, label=y_val_data, reference=i_trn_dt)
        
        # model = lgb.train(cls_param, i_trn_dt, valid_sets=[i_val_dt, i_trn_dt],fobj=wbc_ll,feval=lgb_f1_score,
        #                  num_boost_round=num_iterations, early_stopping_rounds=30, valid_names=['valid','train'])
        
        model = lgb.train(cls_param, i_trn_dt, valid_sets=[i_val_dt, i_trn_dt],fobj=wbc_ll,
                         num_boost_round=num_iterations, early_stopping_rounds=30, valid_names=['valid','train'])
        
        metric_score.append(model.best_score['valid']['auc'])
        
    return metric_score


def wbc_hyopt_obj_func(params, trn_data_df=None):
    '''
    
    '''
    params['num_iterations'] = int(params['num_iterations'])
    
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
    
    # params['importance_type'] = 'gain'
    # params['is_unbalance'] = True
    # params['objective'] = 'binary'
    params['random_state'] = 42
    
    
    metric_score=wbc_cv_train(params, trn_data_df)
        
    score = -np.mean(metric_score) # np.max()

    return {'loss': score, 'status': STATUS_OK}

def wbc_hyopt_train(trn_data_df, param_space):
    '''
    
    '''
    
    obj_func = partial(wbc_hyopt_obj_func, trn_data_df=trn_data_df)
    
    trials = Trials()
    best = fmin(fn=obj_func, 
                space=param_space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials,
                rstate=np.random.RandomState(42))
    return best, trials



def wbc_re_train_final(trn_data_df, opt_params, saved_fname=None):
    '''
    re-train model with optimal hyperparameters and return SHAP values
    '''
    
    cls_param = opt_params.copy()
    
    wbc_alpha = opt_params['wbc_alpha']
    num_iterations = opt_params['num_iterations']
    cls_param['verbose']=-1
    
    del cls_param['wbc_alpha']
    del cls_param['num_iterations']
    
    y_i_bt_data = trn_data_df['new_triage']
    x_i_bt_data = trn_data_df.iloc[:,2:]
    
    wbc_ll = lambda x,y: weighted_log_loss(x, y,wbc_alpha)

    i_trn_dt = lgb.Dataset(x_i_bt_data, label=y_i_bt_data)
    
    model = lgb.train(cls_param,i_trn_dt,fobj=wbc_ll, num_boost_round=num_iterations)
     
            
    return model


def wbc_repeat_perf(trn_data_df, tst_data_df,rand_id_split_fname, opt_params,n_iter=100, saved_fname=None, is_shap=True,feat_ls='all'):
    
    entire_df = pd.concat([trn_data_df, tst_data_df])
    
    with open(rand_id_split_fname, 'rb') as f:
        rand_id_splits = pickle.load(f)
        
    res=[]
    for trn_id_df, tst_id_df in rand_id_splits:
        x_train_df = trn_id_df.merge(entire_df, on=['ind','new_triage'])
        x_test_df = tst_id_df.merge(entire_df, on=['ind','new_triage'])
        
        if feat_ls=='all':
            i_model = wbc_re_train_final(x_train_df, opt_params)
        else:
            feats = ['ind','new_triage'] + feat_ls
            x_train_df = x_train_df[feats].copy()
            x_test_df = x_test_df[feats].copy()
            i_model = wbc_re_train_final(x_train_df, opt_params)
        
        # predict on testing set 
        y_id_label_tst = x_test_df[['ind','new_triage']].copy()
        y_id_label_trn = x_train_df[['ind','new_triage']].copy()
        
        x_tst_df = x_test_df.iloc[:,2:]
        y_pred_tst = i_model.predict(x_tst_df.values)
        y_pred_trn = i_model.predict(x_train_df.iloc[:,2:].values)
        
        y_id_label_tst['y_pred']=sigmoid(y_pred_tst)
        y_id_label_trn['y_pred']=sigmoid(y_pred_trn)
        
        if is_shap:
            shap_values = shap.TreeExplainer(i_model).shap_values(x_tst_df)
        else:
            shap_values=None
    
        res.append((i_model, y_id_label_trn, y_id_label_tst, shap_values))
        
    if saved_fname is not None:
        with open(saved_fname, 'wb') as f:
            pickle.dump(res, f) 
    return res