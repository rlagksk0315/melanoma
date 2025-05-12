#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export results_path='../results/hana/final_main_1_classification'

python main_1.py --results_path "$results_path" \
                               --num_epochs 200 \
                               --learning_rate 0.001