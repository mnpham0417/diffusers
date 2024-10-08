export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/vast/mp5847/diffusers_generated_datasets/van_gogh_5000_sd_v1.4"

accelerate launch --mixed_precision="fp16" train_text_to_image_ascent.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=6 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="/scratch/mp5847/robust-concept-erasure-checkpoints/vg_sd_v1.4_ascent"