
'''


author: Yong Bai on 2021/10/16

'''
import os
import pandas as pd

import argparse

import pickle
import json

import bl_train
import cfdna_covid19


def main(args):
    
    save_root_dir = r'~/model/m2model'
    random_trn_tst_split_fname = r'~/data/random_trn_tst_split.pkl'
    
    print('data preprocessing...')
    trn_df_fanme = os.path.join(save_root_dir, '00_trn_df.csv')
    tst_df_fname = os.path.join(save_root_dir, '00_tst_df.csv')
    
    print('hyopt training...')
    # bl_train.baseline_hy_train(trn_df_fanme, save_root_dir)

    print('retrain with optimal hyperparameters and feature importance...')
    opt_params_fname = os.path.join(save_root_dir, '21_bl_opt_params-auc-false.json')
    # bl_train.baseline_retrain(trn_df_fanme, tst_df_fname, random_trn_tst_split_fname,opt_params_fname, save_root_dir)
    
    # load preprocessed data
    trn_df = pd.read_csv(trn_df_fanme, sep='\t')
    tst_df = pd.read_csv(tst_df_fname, sep='\t')
    
    with open(opt_params_fname,'r') as f:
        opt_params = json.load(f)
    
#     feat_imp_shap_re_df = cfdna_covid19.feat_importance_shap(os.path.join(save_dir, '22_bl_entire_feat_retrain-auc-false.pkl'))
#     feat_imp_shap_re_df.to_csv(os.path.join(save_dir, '23_bl_feature_importance_shap-false.csv'), sep='\t', index=False)

    feat_imp_shap_re_df = pd.read_csv(os.path.join(save_root_dir, '23_bl_feature_importance_shap-false.csv'), sep='\t')
    features_ordered = [str(x) for x in feat_imp_shap_re_df['feat_name'].values]
    
    top_k_f = args.top_k
    # 510 total number of m2model features
    feat_ls = features_ordered[:top_k_f]
    
    save_fanme = os.path.join(save_root_dir, 'bl_perfeat/24_bl_prefeature_performance-false_{}.pkl'.format(top_k_f))
    cfdna_covid19.b_repeat_perf(trn_df,tst_df,random_trn_tst_split_fname, opt_params, saved_fname=save_fanme, is_shap=False, feat_ls=feat_ls)

    print('{} retrain done...'.format(top_k_f)) 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='baseline_m2model')
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=10)

    args = parser.parse_args()
    main(args)