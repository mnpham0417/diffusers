# Robust Concept Erasure

## Van Gogh Experiment (Gradient Ascent)

### Step 1: Generating Images for Concept Erasure

```bash
OUTPUT_DIR="./data/van_gogh"
PROMPT="a painting in the style of Van Gogh"

python3 generate_images.py \
    --output_dir=$OUTPUT_DIR \
    --prompt="$PROMPT" \
    --mode="train" \
    --num_train_images=100
```

### Step 2: Generating Images for Concept Inversion
To make sure that gradient ascent is only preventing Concept Inversion on training images in Step 1, we generate a second set of images for Concept Inversion.
    
```bash
OUTPUT_DIR="./data/van_gogh_ti"
PROMPT="a painting in the style of Van Gogh"

python3 generate_images.py \
    --output_dir=$OUTPUT_DIR \
    --prompt="$PROMPT" \
    --mode="train" \
    --num_train_images=100
```

### Step 3: Performing Concept Erasure

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="./data/van_gogh"
export OUTPUT_DIR="./checkpoint/van_gogh_ascent"

accelerate launch --mixed_precision="fp16" train_text_to_image_ascent.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=20 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=6 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR
```

### Step 4 (Optional): Verifying Concept Erasure

```bash
export MODEL_NAME="./checkpoint/van_gogh_ascent"
export OUTPUT_DIR="./generation/van_gogh_ascent"
export PROMPT="a painting in the style of Van Gogh"

python3 generate_images.py \
    --output_dir=$OUTPUT_DIR \
    --prompt="$PROMPT" \
    --mode="test" \
    --model_path=$MODEL_NAME \
    --num_train_images=10
```

### Step 5: Performing Concept Inversion

```bash
export MODEL_NAME="./checkpoint/van_gogh_ascent"
export DATA_DIR="./data/van_gogh_ti/train"
export OUTPUT_DIR="./checkpoint/van_gogh_ascent_ti"

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
  --output_dir=$OUTPUT_DIR
```

### Step 6 (Optional): Verifying Concept Inversion

```bash
export MODEL_NAME="./checkpoint/van_gogh_ascent"
export MODEL_NAME_TI="./checkpoint/van_gogh_ascent_ti"
export OUTPUT_DIR="./generation/van_gogh_ascent_ti"
export PROMPT="a painting in the style of <art-style>"

python3 generate_images.py \
    --output_dir=$OUTPUT_DIR \
    --prompt="$PROMPT" \
    --mode="test" \
    --model_path=$MODEL_NAME \
    --model_path_ti=$MODEL_NAME_TI \
    --num_train_images=10
```