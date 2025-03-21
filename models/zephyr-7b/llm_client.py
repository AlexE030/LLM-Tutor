from huggingface_hub import InferenceClient

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
        self.instruct_client = InferenceClient(api_key=self.api_key)

    def query_instruct(self, model: str, messages: list, max_tokens: int = 500, temperature: float = 1.0) -> dict:
        """
        Query the LLM using the Instruct model for chat completion.

        Args:
            model (str): The name of the model to query.
            messages (list): A list of messages in the format [{"role": "user", "content": "..."}].
            max_tokens (int): The maximum number of tokens for the response.
            temperature (float): Sampling temperature for response generation.

        Returns:
            dict: The response from the API.
        """
        try:
            response = self.instruct_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Error querying the instruct model API: {e}")