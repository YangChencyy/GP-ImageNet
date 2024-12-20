#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=IM10-7
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/jobs/Imagenet10/im10-7.log

export ID=7
export NF=128

python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=$NF --dset_id $ID --train --eval_train --ood DTD
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=$NF --dset_id $ID --ood SVHN
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=$NF --dset_id $ID --ood LSUN-R
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=$NF --dset_id $ID --ood LSUN-C
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=$NF --dset_id $ID --ood Places365
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=$NF --dset_id $ID --ood iSUN