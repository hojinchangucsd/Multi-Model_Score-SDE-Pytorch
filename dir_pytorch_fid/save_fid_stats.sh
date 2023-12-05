#!/bin/bash

cd ~/Score_SDE/copied_score_sde/

imgs_path=./exp/63M/workdir/mult_500L-500S_eval_folder/ckpt_24/gen_imgs/
stats_path=./assets/fid_stats/500L-500S_stats.npz

python -m pytorch_fid --device cuda:1 --save-stats $imgs_path $stats_path