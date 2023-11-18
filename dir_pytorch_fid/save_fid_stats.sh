#!/bin/bash

cd ~/Score_SDE/copied_score_sde/

imgs_path=./exp/13M/workdir/mult_63M_750-250_eval_folder/ckpt_26/gen_imgs/
stats_path=./assets/fid_stats/mult-7525_stats.npz

python -m pytorch_fid --device cuda:1 --save-stats $imgs_path $stats_path