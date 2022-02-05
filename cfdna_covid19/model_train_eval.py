'''
model train and evaluation

author: Yong Bai on 2021/10/15

'''
import numpy as np
import pandas as pd
import pickle
import os
from functools import partial 
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve,average_precision_score
import shap


from .data_loader import bootstrap_sub_dataset

def get_class_weight(y_label):
    
    from sklearn.utils.class_weight import compute_class_weight
    cls_weight = compute_class_weight('balanced', np.unique(y_label), y_label)
    return dict(zip(np.unique(y_label),cls_weight))


def CV_train(params, trn_data_df, n_fold=5):
    '''
    '''
    
    x_train = trn_data_df.iloc[:,2:].copy().reset_index(drop=True)
    y_train = trn_data_df[['new_triage']].copy().reset_index(drop=True)
    
    kf = StratifiedKFold(n_splits=n_fold) 
    roc_auc=[]
    y_pred_cv = []
    for tr_indx,val_index in kf.split(x_train, y_train):
    
        X_tr_data = x_train.iloc[tr_indx,:]
        X_val_data = x_train.iloc[val_index,:]
        y_tr_data = y_train.iloc[tr_indx,:]
        y_val_data = y_train.loc[val_index,'new_triage'].values
        
        data_tmp = pd.concat([y_tr_data,X_tr_data],axis=1)
        
        all_proba=[]
        for i_bt_data in bootstrap_sub_dataset(data_tmp):
            lgbm = LGBMClassifier(**params)
            
            y_i_bt_data = i_bt_data['new_triage'].values
            x_i_bt_data = i_bt_data.iloc[:,1:]
            
            cls_weight = get_class_weight(y_i_bt_data)
            sam_weight = np.vectorize(cls_weight.get)(y_i_bt_data)
            
            
            lgbm.fit(x_i_bt_data, 
                     y_i_bt_data, 
                     sample_weight=sam_weight,
                     eval_set=[(x_i_bt_data, y_i_bt_data), (X_val_data, y_val_data)],
                     early_stopping_rounds=20, verbose=0)
            
            y_i_pred = lgbm.predict_proba(X_val_data)
            all_proba.append(y_i_pred)
            
        y_pred = (np.mean(all_proba, axis = 0))[:,1]
        
        roc_auc.append(roc_auc_score(y_val_data, y_pred))
        
        y_pred_cv.append((y_val_data,y_pred))
        
    return roc_auc, y_pred_cv
    

def hyopt_obj_func(params, trn_data_df=None):
    '''
    
    '''
    params['importance_type'] = 'gain'
    params['class_weight'] = 'balanced'
    params['objective'] = 'binary'
    params['random_state'] = 42
    
    
    params['n_estimators'] = int(params['n_estimators'])
#     params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['min_child_samples'] = int(params['min_child_samples'])
    
    roc_auc, _=CV_train(params, trn_data_df)
    
    score = -np.mean(roc_auc) # np.max()
    return {'loss': score, 'status': STATUS_OK}


def hyopt_train(trn_data_df, param_space):
    '''
    
    '''
    
    obj_func = partial(hyopt_obj_func, trn_data_df=trn_data_df)
    
    trials = Trials()
    best = fmin(fn=obj_func, 
                space=param_space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials,
                rstate=np.random.RandomState(42))
    return best, trials


def re_train_final_SHAP(trn_data_df, opt_params, saved_fname=None):
    '''
    re-train model with optimal hyperparameters and return SHAP values
    '''
    whole_models = []
    for i_bt_data in bootstrap_sub_dataset(trn_data_df):
        
        lgbm = LGBMClassifier(**opt_params)
        y_i_bt_data = i_bt_data['new_triage']
        x_i_bt_data = i_bt_data.iloc[:,2:]
            
        cls_weight = get_class_weight(y_i_bt_data)
        sam_weight = np.vectorize(cls_weight.get)(y_i_bt_data)
        
        lgbm.fit(x_i_bt_data, 
                 y_i_bt_data, 
                 sample_weight=sam_weight,verbose=0)
        
        # take 1 as positive class: https://github.com/slundberg/shap/issues/526
        shap_values = shap.TreeExplainer(lgbm).shap_values(x_i_bt_data)[1]
        
        whole_models.append((lgbm,shap_values))
    
    if saved_fname is not None:
        with open(saved_fname, 'wb') as f:
            pickle.dump(whole_models, f) 
            
    return whole_models
    

def feat_importance_gain_shap(whole_models):
    '''
    whole_models is by returned re_train_SHAP
    '''
    feat_imp_gain_df = None
    feat_imp_shap_df = None 
    
    for i, i_whole_model in enumerate(whole_models):
        i_model = i_whole_model[0]
        i_shap_values = i_whole_model[1]
        
        # https://github.com/slundberg/shap/issues/632
        mean_abs_shap = np.abs(i_shap_values).mean(0)
        
        if i == 0:
            feat_imp_gain_df = pd.DataFrame({'feat_name': i_model.feature_name_, 
                                        '{}_model_gain'.format(i): i_model.feature_importances_})
            
            feat_imp_shap_df = pd.DataFrame({'feat_name': i_model.feature_name_, 
                                        '{}_model_mean_abs_shap'.format(i): mean_abs_shap})
            
            
        else:
            feat_imp_gain_df = feat_imp_gain_df.merge(pd.DataFrame({'feat_name': i_model.feature_name_, 
                                                          '{}_model_gain'.format(i): i_model.feature_importances_}), 
                                            on='feat_name')
            
            feat_imp_shap_df = feat_imp_shap_df.merge(pd.DataFrame({'feat_name': i_model.feature_name_, 
                                                          '{}_model_mean_abs_shap'.format(i): mean_abs_shap}), 
                                            on='feat_name')      
    
    return feat_imp_gain_df,feat_imp_shap_df
        

def evaluation(tst_data_df, whole_models):
    '''
    whole_models is by returned re_train_SHAP
    '''
    y_tst_array = tst_data_df['new_triage'].values
    x_tst_df = tst_data_df.iloc[:,2:]
    
    y_pred_df = pd.DataFrame({'y_true': y_tst_array})
    for i, i_whole_models in enumerate(whole_models):
        i_model = i_whole_models[0]
        y_i_pred = i_model.predict_proba(x_tst_df)
        y_pred_df['{}_model'.format(i)] = y_i_pred[:,1]
    return y_pred_df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    