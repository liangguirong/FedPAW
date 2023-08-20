#!/bin/bash
#SBATCH -J ms
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH -o /public/home/jd_lgr/ms/FedBeta/output/FedBeta11seed100最新68.txt
#SBATCH -e /public/home/jd_lgr/ms/FedBeta/error/FedBeta11seed100最新68.txt
#SBATCH -p gpu1
cd  $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS
CUDA_VISIBLE_DEVICES=$NP python train_Perl.py
