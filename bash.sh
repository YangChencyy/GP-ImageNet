#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=TinyImageNet10
#SBATCH --nodes=2
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=rivachen@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=standard
#SBATCH --output=/home/rivachen/OOD_Learning_with_GP_boundaries-/cifar10_results.log

# module purge
# conda init bash
source activate GP

python classification.py