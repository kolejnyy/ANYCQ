#!/bin/bash
clear
python train.py --model_dir models/FB15k-237-EFO1/anycq --config configs/model/model_FB15k-237.json --checkpoint_steps 10000 --exp_name Train_ANYCQ_FB15k-237_