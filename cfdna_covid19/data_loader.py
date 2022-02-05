'''
data loader

author: Yong Bai on 2021/10/15

'''

import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split

# dat_root_dir = r'E:\cfDNA\data'
# dat_root_dir = r'/zfssz2/ST_MCHRI/BIGDATA/USER/baiyong/cfnda_covid/new_code/data'


def read_trn_tst_ids(trn_id_fname, tst_id_fname):
    '''
    load training ID and testing ID
    
    return:
    --------
       
    '''

    trn_id_df = pd.read_csv(trn_id_fname)
    tst_id_df = pd.read_csv(tst_id_fname)
    return trn_id_df, tst_id_df


def trn_tst_split(entire_data_df, trn_id_fname, tst_id_fname, on='ind', norm_method='standard'):
    '''
    split entire dataset into training dataset and independent testing dataset
    according to the training ID and testing ID, and normalization.
    
    input:
    --------
    entire_data_df: DataFrame
        contains the columns of "ind" and features
        
    return:
    --------
    trn_data_df: DataFrame
        training dataset: the first two columns are "ind" and "new_triage", followed by normalized features 
    tst_data_df: DataFrame
        independent testing dataset: the first two columns are "ind" and "new_triage", followed by normalized features
    '''
    trn_id_df, tst_id_df = read_trn_tst_ids(trn_id_fname, tst_id_fname)
    trn_data_df = trn_id_df.merge(entire_data_df,on=on)
    tst_data_df = tst_id_df.merge(entire_data_df,on=on)
    
    trn_feat_df = trn_data_df.iloc[:,2:].copy()
    trn_id_re_df = trn_data_df.iloc[:,:2].copy()
    
    tst_feat_df = tst_data_df.iloc[:,2:].copy()
    tst_id_re_df = tst_data_df.iloc[:,:2].copy()
    
    if norm_method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(trn_feat_df)
        trn_feat_df_norm = pd.DataFrame(scaler.transform(trn_feat_df), columns=trn_feat_df.columns)
        tst_feat_df_norm = pd.DataFrame(scaler.transform(tst_feat_df), columns=trn_feat_df.columns)
        
        trn_data_df = pd.concat([trn_id_re_df, trn_feat_df_norm], axis=1)
        tst_data_df = pd.concat([tst_id_re_df, tst_feat_df_norm], axis=1)
    
    return trn_data_df, tst_data_df

    
def bootstrap_sub_dataset(trn_data_df, n_iter=100):
    '''
    bootstrap sampling to balance the training dataset
    
    input:
    --------
    trn_data_df: DataFrame
        training dataset: the first two columns are "ind" and "new_triage", followed by features
        returned feature selection if posiible or trn_tst_split
    
    return:
    --------
    
    '''
#     trn_data_df.drop(columns=['ind'], inplace=True)
    
    pos_data = trn_data_df[trn_data_df.new_triage==1].copy()  # critical
    neg_data = trn_data_df[trn_data_df.new_triage==0].copy()  # noncritical
    
    n_pos = pos_data.shape[0]
#     print('The number of critical patients in training set: {}'.format(n_pos))
#     print('The number of noncritical patients in training set: {}'.format(neg_data.shape[0]))
    from sklearn.utils import resample, shuffle
 
    for i in range(n_iter):  
        bootstrap_samples = resample(neg_data, n_samples=n_pos, replace=True,random_state = i).copy()
        bootstrap_samples.drop_duplicates(inplace=True)
        one_boot = pd.concat([pos_data,bootstrap_samples],axis =0)
        one_boot = shuffle(one_boot,random_state=i) 
        
        yield one_boot


def crt_random_data_split(trn_id_fname, tst_id_fname,n_iter=100, saved_fname=None):
    
    trn_id_df, tst_id_df = read_trn_tst_ids(trn_id_fname, tst_id_fname)
    
    
    entire_id_df = pd.concat([trn_id_df, tst_id_df]).sample(frac=1, random_state=42)
    entire_id_df.reset_index(drop=True, inplace=True)
    
    res=[]
    
    for i_iter in range(n_iter):
        
        x_train_df, x_test_df = train_test_split(entire_id_df,test_size=0.2, 
                                                 stratify=entire_id_df['new_triage'].values, 
                                                 random_state=None,shuffle=True)
    
        res.append((x_train_df, x_test_df))
        
    if saved_fname is not None:
        with open(saved_fname, 'wb') as f:
            pickle.dump(res, f) 

     

    
