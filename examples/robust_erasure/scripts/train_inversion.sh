export MODEL_NAME="/home/mp5847/src/diffusers/examples/robust_erasure/checkpoint/english_springer_sd_v1.4_ascent"
export DATA_DIR="/home/mp5847/src/diffusers/examples/robust_erasure/data/english_springer_ti/train"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token="<art-style>" \
  --initializer_token="art" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=5.0e-03 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --output_dir="/home/mp5847/src/diffusers/examples/robust_erasure/checkpoint/english_springer_sd_v1.4_ascent_ti"