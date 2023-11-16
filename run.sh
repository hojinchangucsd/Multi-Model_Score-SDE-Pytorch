#!/bin/bash

python main.py \
    --config=./configs/ve/13M_cifar10_ncsnpp_continuous.py \
    --eval_folder=mult_63M_500-500_eval_folder \
    --mode=eval \
    --workdir=./exp/13M/workdir/ 