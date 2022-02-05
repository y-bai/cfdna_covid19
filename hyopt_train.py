
'''


author: Yong Bai on 2021/10/06

'''

import os
import numpy as np
import pandas as pd
from hyperopt import hp
import pickle
import json
from tqdm import tqdm

import cfdna_covid19

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def data_prep(dat_root_dir, raw_entire_data_fname,  trn_id_fname, tst_id_fname,
    trn_df_fname, tst_df_fname, feat_sel=False, f_type='lab',min_hit=2, save_dir=None):
    # dat_root_dir = r'/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/cfnda_covid/new_code/data'
    # model_root_dir = r'/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/cfnda_covid/new_code/code/model/lab'
    
    # load raw data
    if f_type=='lab':
        raw_df = pd.read_csv(os.path.join(dat_root_dir, raw_entire_data_fname)).iloc[:,:-1].copy()
        print('lab, drop last col.')
    elif f_type=='tss':
        with open(os.path.join(dat_root_dir, 'tss.fillna.pkl'),'rb') as f:
            raw_df = pickle.load(f)
        raw_df.drop(columns=raw_df.columns[1], inplace=True)
        print('tss, drop second col.')
    else:
        raw_df = pd.read_csv(os.path.join(dat_root_dir, raw_entire_data_fname))
        raw_df.drop(columns=raw_df.columns[1], inplace=True)
        print('{}, drop second col.'.format(f_type))

    # data split and normalization
    trn_df, tst_df = cfdna_covid19.trn_tst_split(raw_df,trn_id_fname, tst_id_fname)

    # feature selection
    if feat_sel:
        feat_sels = ['ind', 'new_triage'] + cfdna_covid19.comb_sel(trn_df, mi_n_neighbors=3, boruta_max_iter=100,
            boruta_perc=90, tentative_sel_rank_thre=5, min_hit=min_hit, min_fraction=0.3, save_dir=save_dir)
        trn_df = trn_df[feat_sels].copy()
        tst_df = tst_df[feat_sels].copy()
    
    trn_df.to_csv(trn_df_fname, sep='\t', index=False)
    tst_df.to_csv(tst_df_fname, sep='\t', index=False)


def hy_train(trn_df_fanme, save_dir):

    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fanme, sep='\t')

    print('total number of features: {}'.format(trn_df.shape[1]))

    # bayes optimization for hyperparameters
    lgb_space = {
        'n_estimators': hp.quniform('n_estimators', 50, 200, 10),  
        #   'num_leaves': hp.choice('num_leaves', np.arange(20, 150, 1, dtype=int)),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.3, 0.8, 0.1),
        'min_split_gain': hp.quniform('min_split_gain', 0.01, 0.1, 0.01), 
        'subsample': hp.uniform('subsample', 0.5, 1.),
        #   'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
        'min_child_samples': hp.quniform('min_child_samples', 5, 100, 5)}
    
    train_out = cfdna_covid19.hyopt_train(trn_df, param_space=lgb_space)

    opt_params = train_out[0]
    opt_params['min_child_samples'] = int(opt_params['min_child_samples'])
    opt_params['n_estimators'] = int(opt_params['n_estimators'])
    opt_params['max_depth'] = int(opt_params['max_depth'])
    opt_params['importance_type'] = 'gain'
    opt_params['class_weight'] = 'balanced'
    opt_params['objective'] = 'binary'
    opt_params['random_state'] = 42

    # save the optimal hyperparameters
    with open(os.path.join(save_dir, '01_opt_params.json'),'w') as f:
        json.dump(opt_params, f)

    # save the cv training results
    _, y_pred_cv = cfdna_covid19.CV_train(opt_params, trn_df)
    with open(os.path.join(save_dir, '02_opt_hyper_cv_y_pred.pkl'), 'wb') as ff:
        pickle.dump(y_pred_cv, ff) 


def retrain_and_feat_importance(trn_df_fanme, opt_params_fname, save_dir, feat_imp='shap'):
    
    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fanme, sep='\t')
    with open(opt_params_fname,'r') as f:
        opt_params = json.load(f)

    # retrain the model with optimal hyperparameters and entire trn_df and save the final model
    whole_models = cfdna_covid19.re_train_final_SHAP(trn_df, opt_params, saved_fname=os.path.join(save_dir, '03_entire_feat_retrain_mean.pkl'))

    # Feature Importance
    # feat_imp_gain_df and feat_imp_shap_df can be used to draw boxplot
    feat_imp_gain_df,feat_imp_shap_df = cfdna_covid19.feat_importance_gain_shap(whole_models)
    feature_importance_gain = pd.DataFrame(list(zip(feat_imp_gain_df['feat_name'].values, feat_imp_gain_df.median(1), feat_imp_gain_df.std(1))),
            columns=['feat_name', 'feat_gain_mean', 'feat_gain_std'])
    feature_importance_gain.sort_values(by=['feat_gain_mean'],ascending=False,inplace=True)

    feature_importance_shap = pd.DataFrame(list(zip(feat_imp_shap_df['feat_name'].values, feat_imp_shap_df.median(1), feat_imp_shap_df.std(1))),
            columns=['feat_name', 'feat_shap_mean', 'feat_shap_std'])
    feature_importance_shap.sort_values(by=['feat_shap_mean'],ascending=False,inplace=True)

    # save all features importance
    feature_importance_shap.to_csv(os.path.join(save_dir,'04_feature_importance_shap.csv'), sep='\t', index=False)
    feature_importance_gain.to_csv(os.path.join(save_dir,'04_feature_importance_gain.csv'), sep='\t', index=False)


def get_perform_on_n_features(trn_df_fname, opt_params_fname, top_k_f, save_dir, feat_imp='shap'):

    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fname, sep='\t')
    with open(opt_params_fname,'r') as f:
        opt_params = json.load(f)

    if feat_imp=='shap':
        feature_importance_df = pd.read_csv(os.path.join(save_dir,'04_feature_importance_shap.csv'), sep='\t')
    else:
        feature_importance_df = pd.read_csv(os.path.join(save_dir,'04_feature_importance_gain.csv'), sep='\t')
    
    feat_name_sorted = feature_importance_df['feat_name'].values.tolist()

    n_feat = len(feat_name_sorted)

    print('total features: {}, current top k = {}'.format(n_feat,top_k_f))

    assert(top_k_f>=1)
    assert(n_feat>=top_k_f)

    cols = ['ind','new_triage']+feat_name_sorted[:top_k_f]
    i_trn_df = trn_df[cols].copy()
    roc_auc, y_pred_cv = cfdna_covid19.CV_train(opt_params, i_trn_df)
    i_res={}
    i_res['auc']=roc_auc
    i_res['y_pred_cv']=y_pred_cv

    with open(os.path.join(save_dir, 'top_f_tmp/10_{}_opt_hyper_feat_n_y_pred.pkl'.format(top_k_f)), 'wb') as ff:
        pickle.dump(i_res, ff)

def combine_featurs_perf(trn_df_fname, opt_params_fname, save_dir, feat_imp='shap'):
    # get performance for all features
    if feat_imp=='shap':
        feature_importance_df = pd.read_csv(os.path.join(save_dir,'04_feature_importance_shap.csv'), sep='\t')
    else:
        feature_importance_df = pd.read_csv(os.path.join(save_dir,'04_feature_importance_gain.csv'), sep='\t')
    
    feat_name_sorted = feature_importance_df['feat_name'].values.tolist()
    
    n_feat = len(feat_name_sorted)

    mean_auc_list = []
    y_pred_cv_list = []

    for k_f in range(1, n_feat+1): 
        k_fname = os.path.join(save_dir, 'top_f_tmp/10_{}_opt_hyper_feat_n_y_pred.pkl'.format(k_f))
        if os.path.exists(k_fname):
            with open(k_fname, 'rb') as f:
                k_f_re = pickle.load(f)        
            mean_auc_list.append((np.mean(k_f_re['auc']),np.std(k_f_re['auc'])))
            y_pred_cv_list.append(k_f_re['y_pred_cv'])
    # print('search optimal number of features....')
    # # save the feature number vs acc
    # mean_auc_list = []
    # y_pred_cv_list = []
    # for i in tqdm(range(n_feat)):
    #     cols = ['ind','new_triage']+feat_name_sorted[:i+1]
    #     i_trn_df = trn_df[cols].copy()
    #     roc_auc, y_pred_cv = cfdna_covid19.CV_train(opt_params, i_trn_df)
    #     mean_auc_list.append((np.mean(roc_auc),np.std(roc_auc)))
    #     y_pred_cv_list.append(y_pred_cv)
    
    with open(os.path.join(save_dir, '05_opt_hyper_feat_n_y_pred.pkl'), 'wb') as ff:
        pickle.dump(y_pred_cv_list, ff)
    
def retrain_w_best_featurs(trn_df_fname, opt_params_fname, top_k_f, save_dir, feat_imp='shap'):
    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fname, sep='\t')
    with open(opt_params_fname,'r') as f:
        opt_params = json.load(f)

    if feat_imp=='shap':
        feature_importance_df = pd.read_csv(os.path.join(save_dir,'04_feature_importance_shap.csv'), sep='\t')
    else:
        feature_importance_df = pd.read_csv(os.path.join(save_dir,'04_feature_importance_gain.csv'), sep='\t')
    
    feat_name_sorted = feature_importance_df['feat_name'].values.tolist()
    final_feats = [str(x) for x in feat_name_sorted[:top_k_f]]
    final_feats_dict = {'n_feats': len(final_feats), 'feats':final_feats}

    with open(os.path.join(save_dir, '06_final_feats.json'),'w') as fff:
        json.dump(final_feats_dict, fff)

    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fname, sep='\t')
    with open(opt_params_fname,'r') as f:
        opt_params = json.load(f)

    final_trn_df = trn_df[['ind','new_triage']+final_feats]
    final_whole_models = cfdna_covid19.re_train_final_SHAP(final_trn_df, opt_params, saved_fname=os.path.join(save_dir, '07_final_feat_retrain_mean.pkl'))
    
    
def weighted_ens(y_true, y_pred_models_ls):
    
    from scipy.optimize import minimize
    from sklearn import metrics
    
    starting_values = [0.5]*len(y_pred_models_ls)
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    bounds = [(0,1)]*len(y_pred_models_ls)
    
    def log_loss_func(weights):
        final_pred_prob = 0
        for weight, i_pred in zip(weights, y_pred_models_ls):
            final_pred_prob += weight * i_pred
        return metrics.log_loss(y_true, final_pred_prob)

    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
    
    # return weight
    return res['x']

def train_roc_pr(y_pred_cv):
    
    from sklearn import metrics
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 500)
    
    precisions = []
    aps = []
    mean_recalls = np.linspace(0, 1, 500)
    
    for i_y_train, i_train_pred in y_pred_cv:
        i_fpr, i_tpr, _ = metrics.roc_curve(i_y_train, i_train_pred) #drop_intermediate=False)
        i_auc = metrics.auc(i_fpr, i_tpr)
        
        interp_tpr = np.interp(mean_fpr, i_fpr, i_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(i_auc)
        
        i_precision, i_recall, _ = metrics.precision_recall_curve(i_y_train, i_train_pred)
        i_average_precision = metrics.average_precision_score(i_y_train, i_train_pred)
        
        interp_precision = np.interp(mean_recalls, i_recall, i_precision,period=np.inf)
        interp_precision[0] = 1.0
        precisions.append(interp_precision)
        aps.append(i_average_precision)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc = np.mean(aucs)#auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    mean_precision = np.mean(precisions, axis=0)
#     mean_precision[-1] = 0.0
    std_precision = np.std(precisions, axis=0)
    
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    
    return mean_tpr,std_tpr,mean_fpr,mean_auc,std_auc, mean_precision, std_precision, mean_recalls, mean_ap, std_ap


def final_eval(tst_df_fname, final_feats_fname, final_models_fname,save_dir):
    '''
    whole_models is by returned re_train_SHAP
    '''

    tst_df = pd.read_csv(tst_df_fname, sep='\t')
    with open(final_models_fname, 'rb') as f:
        whole_models = pickle.load(f)

    with open(final_feats_fname,'r') as ff:
        final_feats = json.load(ff)

    tst_data_df = tst_df[['ind','new_triage']+final_feats['feats']].copy()

    y_tst_array = tst_data_df['new_triage'].values
    x_tst_df = tst_data_df.iloc[:,2:]
    
    y_pred_df = pd.DataFrame({'y_true': y_tst_array})

    for i, i_whole_models in enumerate(whole_models):
        i_model = i_whole_models[0]
        y_i_pred = i_model.predict_proba(x_tst_df)
        y_pred_df['{}_model'.format(i)] = y_i_pred[:,1]
    y_pred_df.to_csv(os.path.join(save_dir, '08_final_tst_pred.csv'), sep='\t', index=False)


    

    












        
        




