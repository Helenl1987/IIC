#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t40:00:00
#SBATCH --mem=10GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jl10005@nyu.edu
#SBATCH --job-name=icc_bs8h_trainAndUnlabelled
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

module purge
source activate icc

# DIR="/scratch/jl10005/data/dc_unsupervised_train"
# ARCH="alexnet"
# LR=0.05
# WD=-5
# K=10
# WORKERS=4
# EXP="/scratch/jl10005/deepcluster/train_output"
# BATCHSIZE=256
#python tools/train.py --config_file extra_scripts/no_unsupervised.yaml
#python tools/train.py --config_file extra_scripts/unsupervised_vgg_a_rotation_stl_10.yaml
#python tools/train.py --config_file extra_scripts/eval_vgg_a_rotation_stl_10.yaml
# python main.py  ${DIR} --exp ${EXP} --batch ${BATCHSIZE} --arch ${ARCH} --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}

DATASET="STL10"
DATASET_ROOT="/scratch/jl10005/data/"
ARCH="ClusterNet5g"
EPOCHS=3200
OUTPUT_K=140
GT_K=10
LR=0.0001
LAMB=1.0
SUBHEADS=5
BATCHSIZE=800
DATALOADERS=2
OUTROOT="/scratch/jl10005/IIC_out"
SAVE_FREQ=3

# 1.2 Semi-supervised overclustering -  figure 6 and supp. mat.
# STL10 (653)
# python -m code.scripts.cluster.cluster_sobel --dataset ${DATASET} \
#     --dataset_root ${DATASET_ROOT} --model_ind 653 --arch ${ARCH} \
#     --num_epochs ${EPOCHS} --output_k ${OUTPUT_K} --gt_k ${GT_K} \
#     --lr ${LR} --lamb ${LAMB} --num_sub_heads ${SUBHEADS} --batch_sz ${BATCHSIZE} \
#     --num_dataloaders ${DATALOADERS} --mix_train --crop_orig --rand_crop_sz 64 \
#     --input_sz 64 --mode IID+ --batchnorm_track --out_root ${OUTROOT} --save_freq ${SAVE_FREQ}

# 1.3 Semi-supervised finetuning - table 3
# Semi-sup overclustering features (650)
python -m code.scripts.cluster.cluster_sobel --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} --model_ind 650 --arch ${ARCH} \
    --num_epochs ${EPOCHS} --output_k ${OUTPUT_K} --gt_k ${GT_K} \
    --lr ${LR} --lamb ${LAMB} --num_sub_heads ${SUBHEADS} --batch_sz ${BATCHSIZE} \
    --num_dataloaders ${DATALOADERS} --mix_train --crop_orig --rand_crop_sz 64 \
    --input_sz 64 --mode IID+ --batchnorm_track --out_root ${OUTROOT} --save_freq ${SAVE_FREQ}
