import os
import logging
from openai import OpenAI
from .openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ChatExternalClient(ChatOpenAI):
    """
    This is the same as engine.openai.ChatOpenAI, but we pass in an external OpenAI client.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    client = None

    def __init__(
        self,
        client: OpenAI,
        model_string: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        **kwargs,
    ):
        """
        :param client: an OpenAI client object.
        :param model_string: the model name, used for the cache file name and chat completion requests.
        :param system_prompt: the system prompt to use in chat completions.

        Example usage with lm-studio local server, but any client that follows the OpenAI API will work.

        ```python
        from openai import OpenAI
        from textgrad.engine.local_model_openai_api import ChatExternalClient

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        engine = ChatExternalClient(client=client, model_string="your-model-name")
        print(engine.generate(max_tokens=40, prompt="What is the meaning of life?"))
        ```

        """

        if os.getenv("OPENAI_API_KEY") is None:
            logger.warning("OPENAI_API_KEY not set. Setting it from client.")
            os.environ["OPENAI_API_KEY"] = client.api_key

        super().__init__(
            model_string=model_string, system_prompt=system_prompt, **kwargs
        )
        self.client = client

class ChatClient(ChatExternalClient):
    def _generate_from_single_prompt(self, prompt: str, system_prompt: str = None, temperature=0, max_tokens=2000, top_p=0.99):
        # return super()._generate_from_single_prompt(prompt, system_prompt, temperature, max_tokens, top_p)
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none
        
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt},
        ]
        extra_body = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        }
        
        response = self.client.chat(
            messages=messages,
            extra_body=extra_body,
        )

        self._save_cache(sys_prompt_arg + prompt, response)
        return response
