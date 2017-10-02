#!/bin/sh

#SBATCH --job-name=pi4p
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=96:00:00

#python just_import.py
python train.py --arch segnet --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size 16
