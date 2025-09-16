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
"""