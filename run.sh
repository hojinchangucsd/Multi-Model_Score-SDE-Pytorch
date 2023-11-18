#!/bin/bash

# t13='./configs/ve/_time_13M.py' \
# t63='./configs/ve/_time_63M.py' \
# t5050='./configs/ve/_time_mult-5050.py' \
# t7525='./configs/ve/_time_mult-7525.py' \
# t9010='./configs/ve/_time_mult-9010.py'

#for config in "$t13" "$t63" "$t5050" "$t7525" "$t9010"; do

    python main.py \
        --config=./configs/ve/13M_cifar10_ncsnpp_continuous.py \
        --eval_folder=mult_63M_900-100_eval_folder \
        --mode=eval \
        --workdir=./exp/13M/workdir/ 

#done