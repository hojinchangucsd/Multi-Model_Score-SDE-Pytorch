#!/bin/bash

python main.py \
                --config=./configs/ve/debug_cifar10_ncsnpp_continuous.py \
                --eval_folder=eval_folder \
                --mode=eval \
                --workdir=./exp/debug/workdir/ 