

import os
import cfdna_covid19

raw_root_dir = r'~/data'

trn_id_fname = os.path.join(raw_root_dir, '399_train_2class_01.csv')
tst_id_fname = os.path.join(raw_root_dir, '399_test_2class_01.csv')
saved_fname = os.path.join(raw_root_dir, 'random_trn_tst_split.pkl')

cfdna_covid19.crt_random_data_split(trn_id_fname, tst_id_fname, saved_fname=saved_fname)