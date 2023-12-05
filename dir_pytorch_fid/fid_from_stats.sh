#!/bin/bash

cd /home/mmorafah@AD.UCSD.EDU/Score_SDE/copied_score_sde/

path1=./assets/fid_stats/cifar10_stats.npz
path2=./assets/fid_stats/500L-500S_stats.npz

python -m pytorch_fid --device cuda:1 "$path1" "$path2"