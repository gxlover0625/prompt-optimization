# change the working directory
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PARENT_DIR"

# set the parameters
model="qwen3-14b_vllm_non-thinking"
dataset="bbh_object_counting"
pipline="textgrad"
output_dir="output"

# run the main script
python textgrad/main.py \
    --pipline $pipline \
    --model $model \
    --dataset $dataset \
    --output_dir $output_dir