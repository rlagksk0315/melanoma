#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export results_path='../results/hana/main_6_classification'

python main_6.py --results_path "$results_path" \
                 --num_epochs 100 \
                 --learning_rate 0.001