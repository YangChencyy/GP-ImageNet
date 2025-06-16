#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=CF100
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/jobs/cifar100.log

python cifar100.py --train --eval_train --ood DTD
python cifar100.py --ood SVHN
python cifar100.py --ood LSUN-R
python cifar100.py --ood LSUN-C
python cifar100.py --ood Places365
python cifar100.py --ood iSUN