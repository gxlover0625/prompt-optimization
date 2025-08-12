from metagpt.ext.spo.components.optimizer import PromptOptimizer
from metagpt.ext.spo.utils.llm_client import SPO_LLM

if __name__ == "__main__":
  # Initialize LLM settings
  SPO_LLM.initialize(
    optimize_kwargs={"model": "Qwen3-14B", "temperature": 0.7},
    evaluate_kwargs={"model": "Qwen3-14B", "temperature": 0.3},
    execute_kwargs={"model": "Qwen3-14B", "temperature": 0}
  )

  # Create and run optimizer
  optimizer = PromptOptimizer(
    optimized_path="workspace",  # Output directory
    initial_round=1,  # Starting round
    max_rounds=10,  # Maximum optimization rounds
    template="GSM8K.yaml",  # Template file
    name="GSM8K",  # Project name
  )

  optimizer.optimize()