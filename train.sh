#!/bin/bash

python molgan_train.py --name molgan_original --model molgan -r 5 --lam 0.50
python conditional_train.py --name molgan_conditional --model molgan -r 5 --lam 0.50 --cond_rate 0.30