#!/usr/bin/env bash

echo "python ../005_m2model_1.py">run5.sh
qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=4g,p=1 run5.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=8g,num_proc=10 -binding linear:10 run5_${m2model_top_k}.sh
# 506: total number of m2model features
# m2model_top_fs=($(echo {1..10}))
# for m2model_top_k in "${m2model_top_fs[@]}"
# do
#     echo "python ../005_m2model.py -k ${m2model_top_k}" > run5_${m2model_top_k}.sh
#     # qsub -cwd -P P18Z10200N0124 -q st.q -l vf=8g,num_proc=10 -binding linear:10 run5_${m2model_top_k}.sh
#     qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=4g,p=1 run5_${m2model_top_k}.sh
# done

# python 005_m2model.py>out.1-199.file & # 5581
# python 005_m2model.py>out.200-399.file & # 6198
# nohup python 005_m2model.py>out.400-506.file 2>&1 & # 6481
# ps -ef | grep 005_m2model.py
