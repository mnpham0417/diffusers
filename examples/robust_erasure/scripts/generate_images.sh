MODEL_NAME="./checkpoint/vg_sd_v1.4_ascent"
OUTPUT_DIR="./generation/van_gogh_ascent"
PROMPT="a painting in the style of Van Gogh"
NUM_TRAIN_IMAGES=10

python3 generate_images.py \
    --output_dir=$OUTPUT_DIR \
    --prompt="$PROMPT" \
    --mode="test" \
    --model_path=$MODEL_NAME \
    --num_train_images=$NUM_TRAIN_IMAGES

# OUTPUT_DIR="./generation/monet_sd"
# PROMPT="A painting in the style of Monet"
# NUM_TRAIN_IMAGES=10

# python3 generate_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode test \
#     --num_train_images $NUM_TRAIN_IMAGES

# OUTPUT_DIR="./generation/van_gogh_ti"
# PROMPT="a painting in the style of <art-style>"
# NUM_TRAIN_IMAGES=10

# python3 generate_images.py \
#     --output_dir $OUTPUT_DIR \
#     --prompt "$PROMPT" \
#     --mode test \
#     --model_path "/home/mp5847/src/diffusers/examples/robust_erasure/checkpoint/vg_sd_v1.4_ascent" \
#     --model_path_ti "/home/mp5847/src/diffusers/examples/robust_erasure/checkpoint/vg_sd_v1.4_ascent_ti" \
#     --num_train_images $NUM_TRAIN_IMAGES