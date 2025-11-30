#!/bin/bash
# 改进的DRL训练脚本
# 针对容量提升的优化配置

python experiments/train_drl.py \
    --N 4 --M 4 \
    --Lt 5 --Lr 5 \
    --SNR_dB 25.0 \
    --A_lambda 3.0 \
    --max_steps 100 \
    --num_episodes 10000 \
    --lr_actor 3e-4 \
    --lr_critic 3e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_epsilon 0.2 \
    --ppo_epochs 10 \
    --batch_size 64 \
    --entropy_coef 0.01 \
    --min_entropy_coef 0.001 \
    --rollout_episodes 4 \
    --lr_anneal \
    --min_lr_factor 0.1 \
    --eval_interval 100 \
    --eval_episodes 20 \
    --eval_seed 2024 \
    --save_interval 500 \
    --seed 42 \
    --device auto \
    --save_dir results/drl_training \
    --use_wandb \
    --wandb_project mimo

