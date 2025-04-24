#!/bin/bash

export results_path='../results/hana/main_8_classification'

python main_8.py --results_path "$results_path" \
                 --num_epochs 100 \
                 --learning_rate 0.001