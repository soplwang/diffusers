#!/bin/sh

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./pokemon"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export ACCELERATE_USE_MPS_DEVICE=true

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --hub_model_id=${HUB_MODEL_ID} \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=1337 \
  "$@"