#!/bin/bash
clear
python train.py --model_dir models/NELL/anycq --config configs/model/model_NELL.json --checkpoint_steps 10000 --exp_name Train_ANYCQ_NELL_