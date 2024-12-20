python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --train --eval_train --ood DTD --tag oneepoch
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --ood SVHN --tag oneepoch
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --ood LSUN-R --tag oneepoch
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --ood LSUN-C --tag oneepoch
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --ood Places365 --tag oneepoch
python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --ood iSUN --tag oneepoch
# python imagenet.py --lr=0.1 --num_classes=10 --bsz=256 --n_features=32 --ood DTD