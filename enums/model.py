from enum import Enum

class Model(Enum):
    FLAN = "http://flan_t5_api:8000/process/"
    DISTILBERT = "http://distillbert_api:8000/process/"
    GBERT = "http://gbert_api:8000/process/"
    LLAMA = "http://llama_api:8000/process/"
    ZEPHYR = "http://zephyr_api:8000/process/"
    BLOOM = "http://bloom_api:8000/process/"
    MISTRAL = "http://mistral_api:8000/process/"
