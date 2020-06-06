#!/bin/bash

python molgan_train.py --name molgan_original --model molgan -r 5 --lam 0.5
python conditional_train.py --name molgan_conditional --model molgan -r 5 --lam 0.5