# change the working directory
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PARENT_DIR"

# set the parameters
model="qwen3-14b_vllm_non-thinking"
dataset="gsm8k"
pipline="promptbreeder"
output_dir="output"
num_mutation_prompts=2
num_thinking_styles=4
num_evals=40
simulations=10
problem="Solve the math word problem, giving your answer as an arabic numeral."

# run the main script
python promptbreeder/main.py \
    --pipline $pipline \
    --model $model \
    --dataset $dataset \
    --output_dir $output_dir \
    --num_mutation_prompts $num_mutation_prompts \
    --num_thinking_styles $num_thinking_styles \
    --num_evals $num_evals \
    --simulations $simulations \
    --problem "$problem"