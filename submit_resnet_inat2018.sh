#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=8
#SBATCH --error=/scratch/ssd004/scratch/kkasa/results/inat2019_resnet50_b/%j_0_log.err
#SBATCH --exclude=gpu179
#SBATCH --gpus-per-node=4
#SBATCH --job-name=timm_inat
#SBATCH --mem=160GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=/scratch/ssd004/scratch/kkasa/results/inat2019_resnet50_b/%j_0_log.out
#SBATCH --open-mode=append
#SBATCH --partition=a40
#SBATCH --qos=m
#SBATCH --time 10:00:00

# activate environment
source ~/venvs/cuda11/bin/activate

# launch
./distributed_train.sh 4 --inat-cat name --data-dir /datasets/inat_comp/2019/ --dataset torch/inat --train-split 2019 --val-split 2019 --num-classes 1010 --mixup 0.0 --mixup-prob 0.0 --aa rand-m8-mstd0.5 --cutmix 0.0 --color-jitter 0.0 --reprob 0.3 --recount 3  --model resnet50 --sched cosine --epochs 100 --warmup-epochs 5 --warmup-lr 1e-6 --lr 1e-3 --reprob 0.0 --smoothing 0.0 --drop 0.0 --batch-size 512 --weight-decay 0.02 --opt lamb --mixup 0.0 --output /scratch/ssd004/scratch/kkasa/results/inat2019_resnet50_b --amp --pretrained --log-wandb --experiment inat2019_resnet50_b

#  --resume /scratch/ssd004/scratch/kkasa/results/inat2019_resnet50_b/inat2019_resnet50_b/last.pth.tar
