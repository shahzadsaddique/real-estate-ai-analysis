"""
Embedding service for generating text embeddings using Vertex AI.

This module provides async methods for generating embeddings with caching,
batch processing, and proper error handling.
"""

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional

from google.api_core import retry
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using Vertex AI."""

    def __init__(self):
        """Initialize embedding service."""
        self._model: Optional[TextEmbeddingModel] = None
        self._cache: Dict[str, List[float]] = {}
        # Try different model versions - some may not be available in all regions/projects
        # Options: textembedding-gecko@001, textembedding-gecko@002, textembedding-gecko@003
        # Or newer: text-embedding-004, text-embedding-005
        self._model_name: str = "text-embedding-005"
        self._initialized: bool = False

    def initialize(self):
        """Initialize Vertex AI and embedding model."""
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=settings.gcp_project_id,
                location=settings.gcp_region,
            )

            # Initialize embedding model
            self._model = TextEmbeddingModel.from_pretrained(self._model_name)

            self._initialized = True
            logger.info(
                f"Embedding service initialized with model: {self._model_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize embedding service: {str(e)}. "
                "Embedding generation will fail at runtime. "
                "To fix: Enable Vertex AI API and set quota project. "
                "See: https://cloud.google.com/docs/authentication/adc-troubleshooting/user-creds"
            )
            # Don't raise - allow worker to start, but embeddings will fail later
            self._initialized = False

    @property
    def model(self) -> TextEmbeddingModel:
        """Get embedding model."""
        if not self._model:
            raise RuntimeError(
                "Embedding model not initialized. Call initialize() first."
            )
        return self._model

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.

        Args:
            text: Input text

        Returns:
            SHA256 hash of text
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache if available.

        Args:
            text: Input text

        Returns:
            Cached embedding or None
        """
        cache_key = self._get_cache_key(text)
        return self._cache.get(cache_key)

    def _add_to_cache(self, text: str, embedding: List[float]):
        """
        Add embedding to cache.

        Args:
            text: Input text
            embedding: Generated embedding
        """
        cache_key = self._get_cache_key(text)
        self._cache[cache_key] = embedding

    def _clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def _generate_embeddings_batch_sync(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (synchronous).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        if not self._initialized:
            self.initialize()

        try:
            # Vertex AI TextEmbeddingModel.get_embeddings is synchronous
            # Returns TextEmbedding objects, need to extract values
            embedding_objects = self.model.get_embeddings(texts)

            # Convert TextEmbedding objects to lists of floats
            # Vertex AI returns TextEmbeddingResponse objects with .values attribute
            embeddings = []
            for emb_obj in embedding_objects:
                try:
                    if hasattr(emb_obj, 'values'):
                        # TextEmbeddingResponse object - extract .values (which should be a list)
                        emb_values = emb_obj.values
                        # Ensure it's a list of floats
                        if isinstance(emb_values, list):
                            # Verify it contains numbers
                            if len(emb_values) > 0 and isinstance(emb_values[0], (int, float)):
                                embeddings.append(emb_values)
                            else:
                                # Might be nested, flatten it
                                embeddings.append([float(v) for v in emb_values])
                        else:
                            # Convert to list
                            embeddings.append(list(emb_values))
                    elif isinstance(emb_obj, list):
                        # Already a list
                        embeddings.append(emb_obj)
                    elif hasattr(emb_obj, '__iter__') and not isinstance(emb_obj, str):
                        # Iterable but not a string - convert to list
                        embeddings.append(list(emb_obj))
                    else:
                        # Try direct conversion
                        embeddings.append(list(emb_obj))
                except Exception as e:
                    logger.error(
                        f"Failed to convert embedding object {type(emb_obj)} to list: {str(e)}"
                    )
                    raise ValueError(
                        f"Cannot convert embedding object to list: {type(emb_obj)}. "
                        f"Object attributes: {dir(emb_obj) if hasattr(emb_obj, '__dict__') else 'N/A'}"
                    ) from e

            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    @retry.Retry(
        predicate=retry.if_exception_type(
            Exception
        ),  # Retry on any exception
        initial=1.0,  # Initial delay in seconds
        maximum=60.0,  # Maximum delay in seconds
        multiplier=2.0,  # Exponential backoff multiplier
        deadline=300.0,  # Maximum total time in seconds
    )
    async def _generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with retry logic (async wrapper).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails after retries
        """
        # Run synchronous embedding generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._generate_embeddings_batch_sync, texts
        )
        return embeddings

    async def generate_embeddings(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with caching and batching.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching (default: True)

        Returns:
            List of embedding vectors (same order as input texts)

        Raises:
            RuntimeError: If model is not initialized
            Exception: If embedding generation fails
        """
        if not self._initialized:
            self.initialize()

        if not texts:
            return []

        # Check cache for each text
        cached_embeddings: Dict[int, List[float]] = {}
        texts_to_embed: List[tuple[int, str]] = []

        if use_cache:
            for idx, text in enumerate(texts):
                cached = self._get_from_cache(text)
                if cached:
                    cached_embeddings[idx] = cached
                else:
                    texts_to_embed.append((idx, text))
        else:
            texts_to_embed = [(idx, text) for idx, text in enumerate(texts)]

        # Generate embeddings for uncached texts
        if texts_to_embed:
            # Extract just the texts for API call
            texts_for_api = [text for _, text in texts_to_embed]

            # Vertex AI has batch size limits (typically 5-100 texts per batch)
            # Process in batches of 50
            batch_size = 50
            new_embeddings: Dict[int, List[float]] = {}

            for batch_start in range(0, len(texts_for_api), batch_size):
                batch_end = min(batch_start + batch_size, len(texts_for_api))
                batch_texts = texts_for_api[batch_start:batch_end]
                batch_indices = [
                    texts_to_embed[i][0]
                    for i in range(batch_start, batch_end)
                ]

                try:
                    # Generate embeddings for this batch
                    batch_embeddings = await self._generate_embeddings_batch(
                        batch_texts
                    )

                    # Map embeddings back to original indices
                    for local_idx, original_idx in enumerate(batch_indices):
                        embedding = batch_embeddings[local_idx]
                        new_embeddings[original_idx] = embedding

                        # Add to cache
                        if use_cache:
                            text = texts[original_idx]
                            self._add_to_cache(text, embedding)

                except Exception as e:
                    logger.error(
                        f"Failed to generate embeddings for batch "
                        f"{batch_start}-{batch_end}: {str(e)}"
                    )
                    # Continue with other batches, but log the error
                    # You might want to raise here depending on requirements
                    raise

            # Combine cached and new embeddings
            all_embeddings = []
            for idx in range(len(texts)):
                if idx in cached_embeddings:
                    all_embeddings.append(cached_embeddings[idx])
                elif idx in new_embeddings:
                    all_embeddings.append(new_embeddings[idx])
                else:
                    # This shouldn't happen, but handle gracefully
                    logger.warning(f"Missing embedding for text at index {idx}")
                    all_embeddings.append([])  # Empty embedding as fallback
        else:
            # All embeddings were cached
            all_embeddings = [cached_embeddings[i] for i in range(len(texts))]

        logger.info(
            f"Generated {len(texts_to_embed)} new embeddings, "
            f"used {len(cached_embeddings)} cached embeddings"
        )

        return all_embeddings

    async def generate_embedding(
        self, text: str, use_cache: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use caching (default: True)

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If model is not initialized
            Exception: If embedding generation fails
        """
        embeddings = await self.generate_embeddings([text], use_cache=use_cache)
        return embeddings[0] if embeddings else []

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._cache),
            "cache_keys": len(self._cache),
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._clear_cache()


# Global service instance
embedding_service = EmbeddingService()
