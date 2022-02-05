'''
feature selection

author: Yong Bai on 2021/10/15

'''

import numpy as np
import pandas as pd

import os

from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV, RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC

from boruta import BorutaPy
from functools import partial 
from collections import OrderedDict
import pickle


def mi_sel(trn_data_df, cat_var='auto', mi_n_neighbors=3):
    '''
    feature selection by mutual_info_classif
    
    input:
    --------
    trn_data_df: DataFrame
        training dataset: the first two columns are "ind" and "new_triage", followed by features
        returned by trn_tst_split() function
    
    '''
    
    trn_feat_df = trn_data_df.iloc[:,2:].copy()
    trn_id_df = trn_data_df.iloc[:,:2].copy()  # ind, new_triage
    y = trn_id_df['new_triage'].values
    
    feat_ori_list = list(trn_feat_df.columns)
    
    
    mi_clf = partial(mutual_info_classif, discrete_features=cat_var,random_state=42, n_neighbors=mi_n_neighbors)
    selector = SelectKBest(mi_clf,k='all').fit(trn_feat_df, y)
    
    # return features and corresponding mi scores in descreasing order
    feat_mi_score = dict(zip(feat_ori_list, selector.scores_))
    return OrderedDict(sorted(feat_mi_score.items(), key=lambda item: item[1],reverse=True))


def boruta_sel(trn_data_df, max_iter=100, perc=100):
    '''
    feature selection by boruta algorithm
    
    input:
    --------
    trn_data_df: DataFrame
        training dataset: the first two columns are "ind" and "new_triage", followed by features
        returned by trn_tst_split() function
    perc: int, float    
        With perc = 100 a threshold is specified. The lower the threshold the more features will be selected. 
    
    '''
    
    trn_feat_df = trn_data_df.iloc[:,2:].copy()
    trn_id_df = trn_data_df.iloc[:,:2].copy()  # ind, new_triage
    y = trn_id_df['new_triage'].values
    
    feat_ori_list = list(trn_feat_df.columns)
    
    rf = RandomForestClassifier(n_jobs=-1, 
                                class_weight= 'balanced', 
                                max_depth=5, # 5 is optimal
                                random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=max_iter, perc=perc)
    feat_selector.fit(trn_feat_df.values, y)
    
    feat_score = dict(zip(feat_ori_list, feat_selector.ranking_))
    
    return OrderedDict(sorted(feat_score.items(), key=lambda item: item[1]))


def rfe_sel(trn_data_df):
    '''
    feature selection by boruta algorithm
    
    input:
    --------
    trn_data_df: DataFrame
        training dataset: the first two columns are "ind" and "new_triage", followed by features
        returned by trn_tst_split() function
    
    '''
    
    trn_feat_df = trn_data_df.iloc[:,2:].copy()
    trn_id_df = trn_data_df.iloc[:,:2].copy()  # ind, new_triage
    y = trn_id_df['new_triage'].values
    
    feat_ori_list = list(trn_feat_df.columns)
    
    et = ExtraTreesClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)  # 5 is optimal
    
    selector = RFECV(et, scoring='f1', n_jobs=-1, cv=3).fit(trn_feat_df, y)
    
    feat_score = dict(zip(feat_ori_list, selector.ranking_))
    
    return OrderedDict(sorted(feat_score.items(), key=lambda item: item[1]))



def comb_sel(trn_data_df,  
             mi_n_neighbors=3, 
             boruta_max_iter=100,
             boruta_perc=90,
             tentative_sel_rank_thre=5, min_hit=2, min_fraction=0.3, save_dir = None):
    """  
    combine feature selection method
    
    
    input:
    --------
    min_hit: None or int
        majority voting if not None
    """

    print('MI feature selection...')
    if save_dir is not None:
        mi_re_fname = os.path.join(save_dir, '00_mi_fel_sel.csv')
        if os.path.exists(mi_re_fname):
            mi_feats_df = pd.read_csv(mi_re_fname, sep='\t')
        else:
            mi_sel_dict = mi_sel(trn_data_df, cat_var=False, mi_n_neighbors=mi_n_neighbors)
            mi_feats_df = pd.DataFrame({'feats':mi_sel_dict.keys(),'mi_ranks':mi_sel_dict.values()})
            mi_feats_df.to_csv(mi_re_fname, sep='\t', index=False)
    else:
        mi_sel_dict = mi_sel(trn_data_df, cat_var=False, mi_n_neighbors=mi_n_neighbors)
        mi_feats_df = pd.DataFrame({'feats':mi_sel_dict.keys(),'mi_ranks':mi_sel_dict.values()})

    mi_feats_df.loc[mi_feats_df['mi_ranks']>0,'mi_ranks']=1
    mi_feats_df.loc[mi_feats_df['mi_ranks']<=0,'mi_ranks']=0
    mi_feats_df['mi_ranks'] = mi_feats_df['mi_ranks'].astype(int)

    print('Boruta feature selection...')
    if save_dir is not None:
        bo_re_fname = os.path.join(save_dir, '00_bo_fel_sel.csv')
        if os.path.exists(bo_re_fname):
            bo_feats_df = pd.read_csv(bo_re_fname, sep='\t')
        else:
            bo_sel_dict = boruta_sel(trn_data_df, max_iter=boruta_max_iter, perc=boruta_perc)
            bo_feats_df = pd.DataFrame({'feats':bo_sel_dict.keys(),'bo_ranks':bo_sel_dict.values()})
            bo_feats_df.to_csv(bo_re_fname, sep='\t', index=False)
    else:
        bo_sel_dict = boruta_sel(trn_data_df, max_iter=boruta_max_iter, perc=boruta_perc)
        bo_feats_df = pd.DataFrame({'feats':bo_sel_dict.keys(),'bo_ranks':bo_sel_dict.values()})

    bo_feats_df.loc[bo_feats_df['bo_ranks']>tentative_sel_rank_thre,'bo_ranks']=0
    bo_feats_df.loc[(2<=bo_feats_df['bo_ranks']) & (bo_feats_df['bo_ranks']<=tentative_sel_rank_thre),'bo_ranks']=1 # tentative selected
    
    print('RFE feature selection...')
    if save_dir is not None:
        rfe_re_fname = os.path.join(save_dir, '00_rfe_fel_sel.csv')
        if os.path.exists(rfe_re_fname):
            rfe_feats_df = pd.read_csv(rfe_re_fname, sep='\t')
        else:
            rfe_sel_dict = rfe_sel(trn_data_df)
            rfe_feats_df = pd.DataFrame({'feats':rfe_sel_dict.keys(),'rfe_ranks':rfe_sel_dict.values()})
            rfe_feats_df.to_csv(rfe_re_fname, sep='\t', index=False)
    else:
        rfe_sel_dict = rfe_sel(trn_data_df)
        rfe_feats_df = pd.DataFrame({'feats':rfe_sel_dict.keys(),'rfe_ranks':rfe_sel_dict.values()})
    
    rfe_feats_df.loc[rfe_feats_df['rfe_ranks']>tentative_sel_rank_thre,'rfe_ranks']=0
    rfe_feats_df.loc[(2<=rfe_feats_df['rfe_ranks']) & (rfe_feats_df['rfe_ranks']<=tentative_sel_rank_thre),'rfe_ranks']=1 # tentative selected

    comb_df = mi_feats_df.merge(bo_feats_df, on='feats').merge(rfe_feats_df, on='feats')
    comb_df['overall_rank']=comb_df.sum(axis=1)
    comb_df.sort_values(by='overall_rank', inplace=True, ascending=False, ignore_index=True)
    
    t_feats = comb_df.shape[0]
    print('total number of features: {}'.format(t_feats))
    
    if min_hit is not None:
        feat_sels = comb_df.loc[comb_df['overall_rank']>=min_hit,'feats'].values 
        print('Majority voting with min_hit = {}, returning {} features'.format(min_hit, len(feat_sels)))
        return feat_sels.tolist()
    else:
        n_feat_sel = np.ceil(t_feats * min_fraction)-1
        print('Selecting {} of original features, returning {} features'.format(min_fraction, n_feat_sel))
        return comb_df.loc[:n_feat_sel,'feats'].values.tolist()

    

def feat_importance_shap(re_fname):
    '''
    '''
    feat_imp_shap_df = None 
    
    with open(re_fname, 'rb') as f:
        re_results = pickle.load(f)
    
    for i, i_re in enumerate(re_results):
        
        # https://github.com/slundberg/shap/issues/632
        shap_values_t = i_re[3]
        
        if isinstance(shap_values_t,list):
            i_shap_values = i_re[3][0] 
        else:
            i_shap_values = i_re[3]
            
        mean_abs_shap = np.abs(i_shap_values).mean(0)
        
        if i == 0:       
            feat_imp_shap_df = pd.DataFrame({'feat_name': i_re[0].feature_name(), 
                                             '{}_mean_abs_shap'.format(i): mean_abs_shap})
            
        else:   
            feat_imp_shap_df = feat_imp_shap_df.merge(pd.DataFrame({'feat_name': i_re[0].feature_name(), 
                                                                    '{}_mean_abs_shap'.format(i): mean_abs_shap}), 
                                                      on='feat_name')
    
    shap_mean = feat_imp_shap_df.iloc[:,1:].mean(1)
    shap_std = feat_imp_shap_df.iloc[:,1:].std(1)
    feat_imp_shap_re_df = feat_imp_shap_df.copy()
    
    feat_imp_shap_re_df['feat_shap_mean'] = shap_mean
    feat_imp_shap_re_df['feat_shap_std'] = shap_std
    
    feat_imp_shap_re_df.sort_values(by=['feat_shap_mean'],ascending=False,inplace=True)
    
    return feat_imp_shap_re_df
        
    
