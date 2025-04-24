#!/bin/bash

export gen_images_path='../generated_images/resnet_cgan/FINAL'
export results_path='../results/resnet_cgan/FINAL'

python training_res_cgan.py --gen_images_path "$gen_images_path" \
                            --results_path "$results_path" \
                            --num_epochs 200 \
                            --learning_rate 0.0002 \
                            --scheduler 'none' \
                            --optimizer 'adam'