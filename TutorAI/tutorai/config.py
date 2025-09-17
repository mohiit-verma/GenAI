
from TutorAI.tutorai.llm import LLMConfig

from typing import Dict
DEFAULT_BOOK_URL = "https://github.com/infoalpha/Data-Science-books/blob/master/storytelling-with-data-cole-nussbaumer-knaflic.pdf"

DEFAULT_LLM_CATALOG: Dict[str, LLMConfig] = {
    # Keep this list small & flexible for the POC; add more as you like.
    # Pick models you actually have resources to run on Colab.
    "mistral-7b-instruct": LLMConfig(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3", dtype="bfloat16"
    ),
    "llama-3-8b-instruct": LLMConfig(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct", dtype="bfloat16"
    ),
    "gemma-2-2b-it": LLMConfig(
        repo_id="google/gemma-2-2b-it", dtype="bfloat16"
    ),
    "phi-3-mini-4k-instruct": LLMConfig(
        repo_id="microsoft/Phi-3-mini-4k-instruct", dtype="bfloat16"
    ),
    "Qwen2.5-0.5B-Instruct": LLMConfig(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16"
    ),
}