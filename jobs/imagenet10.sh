#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=IM10
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/jobs/im10-test.log

# python imagenet.py --lr=0.5 --num_classes=10 --bsz=256 --n_features=64 --train --eval_train
python imagenet.py --lr=0.5 --num_classes=10 --bsz=256 --n_features=64