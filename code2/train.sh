#!/bin/bash

export gen_images_path='../generated_images/isalis/validation2'
export results_path='../results/isalis2'

python training.py --gen_images_path "$gen_images_path" \
                   --results_path "$results_path" \
                   --num_epochs 100 \
                   --learning_rate 0.0001