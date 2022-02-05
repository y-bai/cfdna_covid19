
'''


author: Yong Bai on 2021/10/16

'''
import os

import bl_train


save_root_dir = r'~/model/fragl'
random_trn_tst_split_fname = r'~/data/random_trn_tst_split.pkl'
print('data preprocessing...')
trn_df_fanme = os.path.join(save_root_dir, '00_trn_df.csv')
tst_df_fname = os.path.join(save_root_dir, '00_tst_df.csv')

print('hyopt training...')
# bl_train.baseline_hy_train(trn_df_fanme, save_root_dir)

print('retrain with optimal hyperparameters and feature importance...')
opt_params_fname = os.path.join(save_root_dir, '21_bl_opt_params-auc-false.json')
# bl_train.baseline_retrain(trn_df_fanme, tst_df_fname,random_trn_tst_split_fname, opt_params_fname, save_root_dir)

print('retrain with optimal hyperparameters and pre feature ...')
bl_train.baseline_retrain_prefeature(trn_df_fanme, tst_df_fname,random_trn_tst_split_fname, opt_params_fname, save_root_dir)

