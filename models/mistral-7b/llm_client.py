from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str):
        """
        Initialize the LLM client using the Hugging Face InferenceClient for Instruct models
        and requests for chat models.

        Args:
            api_key (str): The Hugging Face API key for authentication.
        """
        if not api_key:
            raise ValueError("HF_API_TOKEN must be provided or set in the environment variables.")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.client = OpenAI(
            base_url="https://router.huggingface.co/hf-inference/v1",
            api_key=api_key,
        )

    def query_instruct(self, model: str, message: str, max_tokens: int = 500, temperature: float = 1.0) -> dict:
        """
        Query the LLM using the Instruct model for chat completion.

        Args:
            model (str): The name of the model to query.
            message (str): The message to send.
            max_tokens (int): The maximum number of tokens for the response.
            temperature (float): Sampling temperature for response generation.

        Returns:
            dict: The response from the API.
        """
        messages = [
            {
                "role": "user",
                "content": message,
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message
        except Exception as e:
            raise RuntimeError(f"Error querying the instruct model API: {e}")

