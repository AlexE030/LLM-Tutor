import requests




def query(payload):
    r


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
        self.API_URL = "https://api.friendli.ai/dedicated"

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
        try:
            query = {"inputs": message}

            response = requests.post(self.API_URL, headers=self.headers, json=query)
            return response.json()

        except Exception as e:
            raise RuntimeError(f"Error querying the instruct model API: {e}")

