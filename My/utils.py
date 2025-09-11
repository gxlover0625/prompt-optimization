from client import OpenAIClient

# https://github.com/microsoft/LMOps/blob/e210d2c026b9958617887762400778ace81172e6/prompt_optimization/optimizers.py#L105-L110
def generate_synonyms(prompt: str, client: OpenAIClient, *args, **kwargs) -> str:
    rewriter_prompt = (
        "Generate a variation of the following instruction while keeping the semantic meaning.\n\n"
        f"Input: {prompt}\n\n"
        "Output:"
    )
    new_instructions = client.chat(rewriter_prompt)
    return new_instructions

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent / ".env", override=True)

    client = OpenAIClient(model="kimi-k2-250905")
    new_instructions = generate_synonyms("Solve this question step by step", client)
    print(new_instructions)