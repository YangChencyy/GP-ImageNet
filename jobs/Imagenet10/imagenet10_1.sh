#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=IM10-1
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/GP-ImageNet/jobs/Imagenet10/im10-1.log

export ID=1
export NF=32
export LR=0.5
export NCLS=10
export BSZ=256
export TAG="o1"

python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --train --eval_train --ood DTD --tag $TAG
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood SVHN --tag $TAG
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood LSUN-R --tag $TAG
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood LSUN-C --tag $TAG
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood Places365 --tag $TAG
python imagenet.py --lr=$LR --num_classes=$NCLS --bsz=$BSZ --n_features=$NF --dset_id $ID --ood iSUN --tag $TAG