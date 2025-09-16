error_example_template = """
<{index}> 
The model's input is:
{inputs}

The model's response is: 
{response}

The correct label is: {label}
"""

gradient_template = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_examples_str}

For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.
Please do not directly propose the revised instruction, as this is the task of another optimizer. All reasons should be written within <reasons></reasons>.
"""

revised_prompt_template = """
I'm writing prompts for a language model designed for a task.

My current prompt is:
{cur_prompt}

But this prompt gets the following examples wrong:
{error_examples_str}

Based on these errors, the problems with this prompt and the reasons are:
{gradients}

Based on the above information, please write one new prompts following these guidelines:
1. The new prompts should solve the current prompt's problems.
2. The new prompts should consider the list of prompts and evolve based on the current prompt.
3. Each new prompt should be wrapped with <START> and <END>.

The new prompt is:
"""