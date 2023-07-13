#!/bin/sh

export ACCELERATE_USE_MPS_DEVICE=true

# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export OUTPUT_DIR="./danbooru2022"
# export HUB_MODEL_ID="danbooru2022-lora"
# export DATASET_NAME="animelover/danbooru2022"
# export DATASET_CONFIG_NAME="1-full"

# accelerate launch train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --dataset_config_name=$DATASET_CONFIG_NAME \
#   --dataloader_num_workers=8 \
#   --resolution=512 --center_crop --random_flip \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=15000 \
#   --learning_rate=1e-04 \
#   --max_grad_norm=1 \
#   --lr_scheduler="cosine" --lr_warmup_steps=0 \
#   --output_dir=${OUTPUT_DIR} \
#   --hub_model_id=${HUB_MODEL_ID} \
#   --checkpointing_steps=500 \
#   --validation_prompt="1girl" \
#   --seed=13379 \
#   "$@"

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./lzl"
export HUB_MODEL_ID="lzl-lora"
export TRAIN_DIR="./lzl/dataset"

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
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
  --checkpointing_steps=25 \
  --validation_prompt="1girl" \
  --seed=13355 \
  "$@"