#!/bin/bash

python main.py \
    --config=./configs/ve/cifar10_ncsnpp_continuous.py \
    --workdir=./exp/first_test/workdir/ \
    --mode=train \
    --eval_folder=./exp/first_test/eval_folder/