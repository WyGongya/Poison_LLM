#!/bin/bash
torchrun train.py \
    --model_name_or_path "gpt2-poisoned_5" \
    --data_path ./autodl-tmp/traindata.json \
    --bf16 True \
    --output_dir "./gpt2-poisoned_test/" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    >> training.log