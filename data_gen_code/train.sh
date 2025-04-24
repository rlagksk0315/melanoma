#!/bin/bash

export gen_images_path='../generated_images/unet_cgan'
export results_path='../results/unet_cgan'

python training.py --gen_images_path "$gen_images_path" \
                --results_path "$results_path" \
                --num_epochs 200 \
                --learning_rate 0.0002 \
                --scheduler 'none' \
                --optimizer 'adam'