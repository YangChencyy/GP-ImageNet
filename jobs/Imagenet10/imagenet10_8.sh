#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=IM10-8
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/jobs/Imagenet10/im10-8.log

export ID=8
export NF=128
export LR=0.1
export NCLS=10
export BSZ=256

python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --train --eval_train --ood DTD
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood SVHN
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood LSUN-R
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood LSUN-C
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood Places365
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood iSUN