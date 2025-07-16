# change the working directory
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PARENT_DIR"

# set the parameters
model="qwen3-14b_vllm_non-thinking"
dataset="gsm8k"
pipline="zerocot"
output_dir="output"

# run the main script
python src/main.py \
    --pipline $pipline \
    --model $model \
    --dataset $dataset \
    --output_dir $output_dir