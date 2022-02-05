'''


author: Yong Bai on 2021/10/14

'''

import os
import numpy as np
import pandas as pd
from hyperopt import hp
import pickle
import json

import cfdna_covid19

def fl_hy_train(trn_df_fanme, save_dir):
    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fanme, sep='\t')

    print('total number of features: {}'.format(trn_df.shape[1]))

    # bayes optimization for hyperparameters
    lgb_space = {
        'num_iterations': hp.quniform('num_iterations', 80, 300, 20), # or n_estimators
        
        # the most import feature for lgmb
        # https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        'num_leaves': hp.quniform('num_leaves', 10, 300, 2),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 2, 60, 1), # or min_child_samples
        # 'min_sum_hessian_in_leaf': hp.choice('min_sum_hessian_in_leaf', np.arange(1, 8, 1, dtype=int)), # or min_child_weight
        
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01), 
        'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 1.,0.1), # or subsample
        'feature_fraction':hp.quniform('feature_fraction', 0.3, 0.8, 0.1),  # or colsample_bytree
        
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
        
        'fl_alpha': hp.uniform('fl_alpha', 0.1, 0.75),
        'fl_gamma': hp.uniform('fl_gamma', 0.5, 5)
        
        # 'min_gain_to_split': hp.quniform('min_gain_to_split', 0.1, 5, 0.01), # or min_split_gain
    }
    
    train_out = cfdna_covid19.fl_hyopt_train(trn_df, param_space=lgb_space)

    opt_params = train_out[0]
    opt_params['min_data_in_leaf'] = int(opt_params['min_data_in_leaf'])
    opt_params['max_depth'] = int(opt_params['max_depth'])
    opt_params['num_leaves'] = int(opt_params['num_leaves'])
    opt_params['num_iterations'] = int(opt_params['num_iterations'])
    
    # opt_params['importance_type'] = 'gain'
    # opt_params['is_unbalance'] = True
    # opt_params['objective'] = 'binary'
    opt_params['random_state'] = 42

    # save the optimal hyperparameters
    with open(os.path.join(save_dir, '31_fl_opt_params-auc.json'),'w') as f:
        json.dump(opt_params, f)
        

def fl_retrain(trn_df_fanme, tst_df_fname,rand_id_split_fname, opt_params_fname, save_dir):
    
    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fanme, sep='\t')
    tst_df = pd.read_csv(tst_df_fname, sep='\t')
    
    with open(opt_params_fname,'r') as f:
        opt_params = json.load(f)

    # retrain the model with optimal hyperparameters
    cfdna_covid19.fl_repeat_perf(trn_df,tst_df,rand_id_split_fname, opt_params, 
                                saved_fname=os.path.join(save_dir, '32_fl_entire_feat_retrain-auc.pkl'))
    
    print('retrain done')
    

def fl_retrain_prefeature(trn_df_fanme, tst_df_fname,rand_id_split_fname,opt_params_fname, save_dir):
    
    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fanme, sep='\t')
    tst_df = pd.read_csv(tst_df_fname, sep='\t')
    
    with open(opt_params_fname,'r') as f:
        opt_params = json.load(f)
        
    feat_imp_shap_re_df = cfdna_covid19.feat_importance_shap(os.path.join(save_dir, '32_fl_entire_feat_retrain-auc.pkl'))
    
    feat_imp_shap_re_df.to_csv(os.path.join(save_dir, '33_fl_feature_importance_shap.csv'), sep='\t', index=False)
    
    features_ordered = [str(x) for x in feat_imp_shap_re_df['feat_name'].values]
    
    ress = []
    for feat_idx in range(len(features_ordered)):
        feat_ls = features_ordered[:feat_idx+1]
        i_re = cfdna_covid19.fl_repeat_perf(trn_df,tst_df,rand_id_split_fname, opt_params, 
                                           saved_fname=None, is_shap=False, feat_ls=feat_ls)
        ress.append((feat_ls, i_re))

    with open(os.path.join(save_dir, '34_fl_prefeature_performance.pkl'), 'wb') as f:
            pickle.dump(ress, f) 
    
    print('retrain done')

    