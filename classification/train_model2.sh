#!/bin/bash

export results_path='../results/sav/main_2_classification'

python main_2.py --results_path "$results_path" \
                               --num_epochs 200 \
                               --learning_rate 0.001