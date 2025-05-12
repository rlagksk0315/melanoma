#!/bin/bash

export results_path='../results/hana/final_main_5_classification'

python main_5.py --results_path "$results_path" \
                 --num_epochs 200 \
                 --learning_rate 0.001