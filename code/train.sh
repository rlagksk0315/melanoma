#!/bin/bash

export gen_images_path='../generated_images/lr_0.0002_decay_True_epochs_200'
export results_path='../results/lr_0.0002_decay_True_epochs_200'

python training.py --gen_images_path "$gen_images_path" \
                --results_path "$results_path" \
                --num_epochs 200 \
                --learning_rate 0.0002 \
                --use_lr_decay # get rid of this flag if you don't want to apply learning rate decay