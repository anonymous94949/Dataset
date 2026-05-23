#!/bin/bash

DATA_DIR=/path/to/images
META_DIR=./meta_data
CHECKPOINT=./checkpoints/last.pth.tar

python inference.py \
  --checkpoint ${CHECKPOINT} \
  --data_dir ${DATA_DIR} \
  --meta_dir ${META_DIR} \
  --csv_file test.csv \
  --output_preds best_preds.csv
