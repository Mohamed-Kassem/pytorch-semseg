#!/bin/sh

#SBATCH --job-name=pi4p
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --output=tmp.out
#output=batch_size_1_fcn_segnet.out

#python just_import.py
#python train.py --arch fcn8s --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size 1
#python validate.py --arch segnet --model_path val_segnet_epoch17.pth.tar --dataset pascal --img_rows 256 --img_cols 256 --batch_size 16 --split val
#python validate.py --arch fcn8s --model_path val_fcn8s_epoch11.pth.tar --dataset pascal --img_rows 256 --img_cols 256 --batch_size 16 --split val

python validate.py --arch fcn8s --model_path val_fcn8s_epoch15.pth.tar --dataset pascal --img_rows 256 --img_cols 256 --batch_size 1 --split val --cuda_index 0 > fcn8s-epoch15.out
python validate.py --arch segnet --model_path val_segnet_epoch23.pth.tar --dataset pascal --img_rows 256 --img_cols 256 --batch_size 1 --split val --cuda_index 1 > segnet-epoch18.out

# cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,/ /g')

# for cudaDev in $cudaDevs
# do
#   echo cudaDev = $cudaDev
#   #srun --gres=gpu:tesla:1 -n 1 --exclusive ./gpuMemTest.sh > gpuMemTest.out.$cudaDev 2>&1 &
#   #$cudaMemTest --num_passes 1 --device $cudaDev > gpuMemTest.out.$cudaDev 2>&1 &

#   if [ $cudaDev -eq 0 ]
#   then
#     python train.py --arch fcn8s --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size 1 --cuda_index 0 > batch_size_1_fcn_segnet.out.0 &
#   else
#     python train.py --arch segnet --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size 1 --cuda_index 1 > batch_size_1_fcn_segnet.out.1 &
#   fi
# done

# wait