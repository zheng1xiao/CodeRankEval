import random
from openai import OpenAI

class vllmAPI:
    def __init__(self, model: str, api_base: str, api_key: str = "EMPTY"):
        """
        初始化 vllmAPI 类。
        :param model: 要使用的 LLM 模型名称。
        :param api_base: OpenAI API 的基础 URL。
        :param api_key: API 密钥，默认为 "EMPTY"。
        """
        self.model = model
        self.api_bases = api_base.split(",")
        self.api_keys = api_key.split(",")

    def generate(self, system_prompt: str, user_input: str, temperature: float = 0.0, max_tokens: int = 2048):
        """
        生成 LLM 响应。
        :param system_prompt: 系统指令。
        :param user_input: 用户输入。
        :param temperature: 采样温度，默认 0.3。
        :param max_tokens: 生成的最大 token 数，默认 1024。
        :return: 生成的文本。
        """
        api_key = random.choice(self.api_keys)
        api_base = random.choice(self.api_bases)
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_input}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def generateWithMessage(self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 2048):
        """
        生成 LLM 响应。
        :param system_prompt: 系统指令。
        :param user_input: 用户输入。
        :param temperature: 采样温度，默认 0.3。
        :param max_tokens: 生成的最大 token 数，默认 1024。
        :return: 生成的文本。
        """
        api_key = random.choice(self.api_keys)
        api_base = random.choice(self.api_bases)
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content