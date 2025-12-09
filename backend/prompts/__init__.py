"""
Prompts package.

This package exports all system prompts for LLM orchestration.
"""

from prompts.analysis_prompt import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_PROMPT_TEMPLATE,
)
from prompts.formatting_prompt import (
    JSON_FORMATTING_PROMPT,
    JSON_VALIDATION_USER_PROMPT_TEMPLATE,
)
from prompts.retrieval_prompt import (
    RETRIEVAL_REFINEMENT_PROMPT,
    RETRIEVAL_USER_PROMPT_TEMPLATE,
)

__all__ = [
    # Analysis prompts
    "ANALYSIS_SYSTEM_PROMPT",
    "ANALYSIS_USER_PROMPT_TEMPLATE",
    # Retrieval prompts
    "RETRIEVAL_REFINEMENT_PROMPT",
    "RETRIEVAL_USER_PROMPT_TEMPLATE",
    # Formatting prompts
    "JSON_FORMATTING_PROMPT",
    "JSON_VALIDATION_USER_PROMPT_TEMPLATE",
]
