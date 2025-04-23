#!/bin/bash

export results_path='../results/sav/main_1_classification'

python main_1.py --results_path "$results_path" \
                               --num_epochs 200 \
                               --learning_rate 0.001