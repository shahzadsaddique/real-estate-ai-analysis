"""
LLM service for LangChain orchestration with Vertex AI.

This module provides LLM initialization, prompt template management,
chain composition utilities, and error handling for LangChain workflows.
"""

import logging
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI

from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations using LangChain and Vertex AI."""

    def __init__(self):
        """Initialize LLM service."""
        self._llm: Optional[ChatVertexAI] = None
        self._initialized: bool = False
        self._default_temperature: float = 0.2
        self._default_max_tokens: int = 4096
        self._default_top_p: float = 0.95
        self._default_top_k: int = 40

    def initialize(self):
        """Initialize Vertex AI and LLM model."""
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=settings.gcp_project_id,
                location=settings.gcp_region,
            )

            # Initialize LLM will be done lazily in get_llm()
            self._initialized = True
            logger.info("LLM service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise

    def get_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> ChatVertexAI:
        """
        Get or create LangChain ChatVertexAI instance.

        Args:
            temperature: Sampling temperature (0.0-1.0), lower = more deterministic
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            model_name: Optional model name override

        Returns:
            ChatVertexAI instance
        """
        if not self._initialized:
            self.initialize()

        # Use provided values or defaults
        temp = temperature if temperature is not None else self._default_temperature
        max_toks = max_tokens if max_tokens is not None else self._default_max_tokens
        tp = top_p if top_p is not None else self._default_top_p
        tk = top_k if top_k is not None else self._default_top_k
        model = model_name or settings.vertex_ai_model_name

        # Create new LLM instance with specified parameters
        # Note: We create a new instance each time to allow different parameters
        llm = ChatVertexAI(
            model_name=model,
            temperature=temp,
            max_output_tokens=max_toks,
            top_p=tp,
            top_k=tk,
            project=settings.gcp_project_id,
            location=settings.gcp_region,
        )

        logger.debug(
            f"Created LLM instance: model={model}, "
            f"temperature={temp}, max_tokens={max_toks}"
        )

        return llm

    def create_chain(
        self,
        prompt: PromptTemplate,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        model_name: Optional[str] = None,
        output_key: str = "text",
    ) -> LLMChain:
        """
        Create a LangChain LLMChain with the specified prompt.

        Args:
            prompt: LangChain PromptTemplate
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            model_name: Optional model name override
            output_key: Key for chain output (default: "text")

        Returns:
            LLMChain instance
        """
        llm = self.get_llm(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            model_name=model_name,
        )

        chain = LLMChain(llm=llm, prompt=prompt, output_key=output_key)

        logger.debug(f"Created LLMChain with output_key: {output_key}")
        return chain

    def create_prompt_template(
        self,
        template: str,
        input_variables: List[str],
        template_format: str = "f-string",
    ) -> PromptTemplate:
        """
        Create a LangChain PromptTemplate.

        Args:
            template: Prompt template string
            input_variables: List of variable names in template
            template_format: Template format ("f-string" or "jinja2")

        Returns:
            PromptTemplate instance
        """
        return PromptTemplate(
            template=template,
            input_variables=input_variables,
            template_format=template_format,
        )

    def run_chain(
        self,
        chain: LLMChain,
        inputs: Dict[str, Any],
        use_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a LangChain chain with error handling and fallback.

        Args:
            chain: LLMChain instance to run
            inputs: Input dictionary for the chain
            use_fallback: Whether to use fallback on error

        Returns:
            Chain output dictionary

        Raises:
            Exception: If chain execution fails and no fallback
        """
        try:
            result = chain.invoke(inputs)
            logger.debug(f"Chain executed successfully")
            return result
        except Exception as e:
            logger.error(f"Chain execution failed: {str(e)}")

            if use_fallback:
                logger.info("Attempting fallback with lower temperature")
                # Retry with more deterministic settings
                fallback_llm = self.get_llm(temperature=0.0, max_tokens=2048)
                fallback_chain = LLMChain(
                    llm=fallback_llm,
                    prompt=chain.prompt,
                    output_key=chain.output_key,
                )

                try:
                    result = fallback_chain.invoke(inputs)
                    logger.info("Fallback chain executed successfully")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback chain also failed: {str(fallback_error)}")
                    raise Exception(
                        f"Chain execution failed: {str(e)}. "
                        f"Fallback also failed: {str(fallback_error)}"
                    ) from fallback_error
            else:
                raise

    async def run_chain_async(
        self,
        chain: LLMChain,
        inputs: Dict[str, Any],
        use_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a LangChain chain asynchronously with error handling.

        Args:
            chain: LLMChain instance to run
            inputs: Input dictionary for the chain
            use_fallback: Whether to use fallback on error

        Returns:
            Chain output dictionary

        Raises:
            Exception: If chain execution fails and no fallback
        """
        try:
            result = await chain.ainvoke(inputs)
            logger.debug(f"Chain executed successfully (async)")
            return result
        except Exception as e:
            logger.error(f"Async chain execution failed: {str(e)}")

            if use_fallback:
                logger.info("Attempting async fallback with lower temperature")
                # Retry with more deterministic settings
                fallback_llm = self.get_llm(temperature=0.0, max_tokens=2048)
                fallback_chain = LLMChain(
                    llm=fallback_llm,
                    prompt=chain.prompt,
                    output_key=chain.output_key,
                )

                try:
                    result = await fallback_chain.ainvoke(inputs)
                    logger.info("Async fallback chain executed successfully")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Async fallback chain also failed: {str(fallback_error)}")
                    raise Exception(
                        f"Async chain execution failed: {str(e)}. "
                        f"Fallback also failed: {str(fallback_error)}"
                    ) from fallback_error
            else:
                raise

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default LLM configuration.

        Returns:
            Dictionary with default configuration values
        """
        return {
            "temperature": self._default_temperature,
            "max_tokens": self._default_max_tokens,
            "top_p": self._default_top_p,
            "top_k": self._default_top_k,
            "model_name": settings.vertex_ai_model_name,
        }

    def update_default_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        """
        Update default configuration values.

        Args:
            temperature: New default temperature
            max_tokens: New default max tokens
            top_p: New default top_p
            top_k: New default top_k
        """
        if temperature is not None:
            self._default_temperature = temperature
        if max_tokens is not None:
            self._default_max_tokens = max_tokens
        if top_p is not None:
            self._default_top_p = top_p
        if top_k is not None:
            self._default_top_k = top_k

        logger.info("Updated default LLM configuration")


# Global service instance
llm_service = LLMService()
