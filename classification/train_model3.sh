#!/bin/bash

export results_path='../results/sav/main_3_classification'

python main_3.py --results_path "$results_path" \
                               --num_epochs 1 \
                               --learning_rate 0.001