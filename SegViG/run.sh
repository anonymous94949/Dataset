#!/bin/bash

DATA_DIR=/path/to/images
META_DIR=/path/to/meta_data/swin_large_semantic

python train.py \
  --training_mech vig \
  --model vig_b_224_gelu \
  --pretrained \
  --pretrain_path ./pretrained_model/vig_b_82.6.pth \
  --sched cosine \
  --epochs 50 \
  --opt adamw \
  --batch-size 64 \
  --warmup-lr 1e-6 \
  --model-ema \
  --model-ema-decay 0.99996 \
  --aa rand-m9-mstd0.5-inc1 \
  --color-jitter 0.4 \
  --warmup-epochs 20 \
  --opt-eps 1e-8 \
  --remode pixel \
  --reprob 0.25 \
  --amp \
  --lr 2e-3 \
  --weight-decay 0.05 \
  --drop 0 \
  --drop-path 0.1 \
  --meta_dir ${META_DIR} \
  --no-prefetcher \
  --data_dir ${DATA_DIR} \
  --num-classes 4 \
  --corn_loss \
  --smoothing 0 \
  --use_segmentation_edge knn_in_seg \
  --contrastive_loss \
  --num_knn 18 \
  --overlap \
  --eval-metric top1
