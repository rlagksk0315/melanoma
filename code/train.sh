#!/bin/bash

export gen_images_path='../generated_images/justin_w10.5_noin_res'
export results_path='../results/justin_w10.5_noin_res'

python training_res_cgan.py --gen_images_path "$gen_images_path" \
                --results_path "$results_path" \
                --num_epochs 200 \
                --learning_rate 0.0002 \
                --use_lr_decay # get rid of this flag if you don't want to apply learning rate decay