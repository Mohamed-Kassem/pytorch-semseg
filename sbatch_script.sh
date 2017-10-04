#!/bin/sh

#SBATCH --job-name=pi4p
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --nodelist=comp003
#SBATCH --cpus-per-task=1
#SBATCH --time=256:00:00
#SBATCH --output=%j-tmp.out
#output=batch_size_1_fcn_segnet.out

#python just_import.py
#python train.py --arch fcn8s --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size 1
#python validate.py --arch segnet --model_path val_segnet_epoch17.pth.tar --dataset pascal --img_rows 256 --img_cols 256 --batch_size 16 --split val
#python validate.py --arch fcn8s --model_path val_fcn8s_epoch11.pth.tar --dataset pascal --img_rows 256 --img_cols 256 --batch_size 16 --split val

if [ "$1" == "validate" ]
then
    #VALIDATION
    echo "********** $1 **********"
    FCN="$2"
    export CUDA_VISIBLE_DEVICES=0
    python validate.py --arch fcn8s --model_path "${FCN}.pth.tar" --dataset pascal --img_rows 256 --img_cols 256 --batch_size 1 --split val --cuda_index 0 > "${FCN}.out" 2>&1 &
    SEG="$3"
    export CUDA_VISIBLE_DEVICES=1
    python validate.py --arch segnet --model_path "${SEG}.pth.tar" --dataset pascal --img_rows 256 --img_cols 256 --batch_size 1 --split val --cuda_index 0 > "${SEG}.out" 2>&1 &
elif [ "$1" == "train" ]
then
    # TRAIN
    echo "********** $1 **********"
    BATCH_SIZE=1
    export CUDA_VISIBLE_DEVICES=0
    python train.py --arch fcn8s --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size 1 --exp_index 0 --validate_every 5 --job_id ${SLURM_JOB_ID} --resume fcn8s_1_1e-05_concat_False_model_9.pth.tar > "${SLURM_JOB_ID}_0_batch_size_${BATCH_SIZE}_fcn8s.out" 2>&1 &
    # python train.py --arch segnet --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size ${BATCH_SIZE} --cuda_index 0 --l_rate 1e-4 > "${SLURM_JOB_ID}_lr1e-4_batch_size_${BATCH_SIZE}_segnet.out" 2>&1 &

    export CUDA_VISIBLE_DEVICES=1
    python train.py --arch fcn8s --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size 1 --exp_index 1 --validate_every 5 --kassem --job_id ${SLURM_JOB_ID} --resume fcn8s_1_1e-05_concat_True_model_9.pth.tar > "${SLURM_JOB_ID}_1_batch_size_${BATCH_SIZE}_fcn8s.out" 2>&1 &
    # python train.py --arch segnet --dataset pascal --n_epoch 150 --img_rows 256 --img_cols 256 --batch_size ${BATCH_SIZE} --cuda_index 0 --l_rate 1e-3 > "${SLURM_JOB_ID}_lr1e-3_batch_size_${BATCH_SIZE}_segnet.out" 2>&1 &
else
    echo "$1 is unrecognized input"
fi


wait