#!/usr/bin/env bash

# echo "python ../002_fragl.py">run2.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run2.sh

# echo "python ../003_tss.py">run3.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run3.sh

# calculate performance on top n features 
# 177: total number of tss features
# tss_top_fs_1=($(echo {1..177}))
# tss_top_fs_2=($(echo {1401..2800}))
top_k_feat=($(echo {1..510}))
for top_k in "${top_k_feat[@]}"
do
    echo "python ../105_baseline_m2model.py -k ${top_k}" > run_bl5_${top_k}.sh
    qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_bl5_${top_k}.sh
done
# echo "python ../003_tss.py">run3.sh
# qsub -cwd -P P18Z10200N0124 -q st_gpu1.q@cngb-gpu-e15-1 -l vf=4g,p=1 run3.sh

# echo "python ../004_motif.py">run4.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run4.sh



####################



# echo "python ../101_baseline_lab.py">run_bl_f_1.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_bl_f_1.sh

# echo "python ../102_baseline_fragl.py">run_bl_f_2.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_bl_f_2.sh

# echo "python ../103_baseline_tss.py">run_bl_f_3.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_bl_f_3.sh

# echo "python ../104_baseline_motif.py">run_bl_f_4.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_bl_f_4.sh

# echo "python ../105_baseline_m2model.py">run_bl_f_5.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_bl_f_5.sh

# echo "python ../201_fl_lab.py">run_fl_1.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_fl_1.sh

# echo "python ../202_fl_fragl.py">run_fl_2.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_fl_2.sh

# echo "python ../203_fl_tss.py">run_fl_3.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_fl_3.sh

# echo "python ../204_fl_motif.py">run_fl_4.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_fl_4.sh

# echo "python ../205_fl_m2model.py">run_fl_5.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_fl_5.sh

# echo "python ../301_wbc_lab.py">run_wbc_1.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_wbc_1.sh

# echo "python ../302_wbc_fragl.py">run_wbc_2.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_wbc_2.sh

# echo "python ../303_wbc_tss.py">run_wbc_3.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_wbc_3.sh

# echo "python ../304_wbc_motif.py">run_wbc_4.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_wbc_4.sh

# echo "python ../305_wbc_m2model.py">run_wbc_5.sh
# qsub -cwd -P P18Z10200N0124 -q st.q -l vf=4g,num_proc=10 -binding linear:10 run_wbc_5.sh
