#!/bin/bash

export results_path='../results/hana/main_4_classification'

python main_4.py --results_path "$results_path" \
                 --num_epochs 100 \
                 --learning_rate 0.001