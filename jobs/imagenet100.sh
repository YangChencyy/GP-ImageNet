#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=IM100
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/jobs/im100-1.log

python imagenet.py --lr=0.1 --num_classes=100 --bsz=64 --n_features=256 --tag=1 --train --eval_train
# python imagenet.py --lr=0.1 --num_classes=100 --bsz=64 --n_features=128