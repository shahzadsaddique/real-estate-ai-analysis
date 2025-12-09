"""
Pinecone service for vector database operations.

This module provides async methods for storing and retrieving document embeddings
using Pinecone as a vector database for semantic search.

Uses Pinecone SDK v5+ which supports both pod-based and serverless indexes.
"""

import logging
from typing import Dict, List, Optional

from pinecone import Pinecone

from config import settings
from models.chunk import Chunk

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for Pinecone vector database operations."""

    def __init__(self):
        """Initialize Pinecone service."""
        self._client: Optional[Pinecone] = None
        self._index = None
        self._initialized: bool = False
        self._index_name: str = settings.pinecone_index_name

    def initialize(self):
        """
        Initialize Pinecone client and connect to index.

        Raises:
            Exception: If initialization fails
        """
        # If already initialized, skip
        if self._initialized:
            logger.debug("Pinecone service already initialized, skipping")
            return

        try:
            # Determine if using serverless (HOST) or pod-based (environment)
            is_serverless = bool(settings.pinecone_host)
            connection_info = (
                f"host: {settings.pinecone_host}" if is_serverless
                else f"environment: {settings.pinecone_environment or settings.pinecone_region}"
            )
            
            logger.info(
                f"Initializing Pinecone service (index: {self._index_name}, {connection_info})"
            )
            
            # Validate configuration
            if not settings.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY is not set in configuration")
            
            # Initialize Pinecone client (new SDK v5+ API)
            self._client = Pinecone(api_key=settings.pinecone_api_key)
            logger.debug("Pinecone client initialized")

            # Connect to index
            if is_serverless:
                # Serverless index - use host URL directly
                if not settings.pinecone_host:
                    raise ValueError("HOST is required for serverless indexes")
                
                logger.info(
                    f"Connecting to serverless index '{self._index_name}' "
                    f"using host: {settings.pinecone_host}"
                )
                
                # For serverless indexes, connect using the host URL
                # Remove https:// prefix if present
                host = settings.pinecone_host.replace("https://", "").replace("http://", "")
                
                try:
                    self._index = self._client.Index(host=host)
                    logger.debug(f"Connected to index using host: {host}")
                except Exception as host_error:
                    # If host parameter doesn't work, try with name
                    logger.debug(f"Host connection failed, trying with index name: {str(host_error)}")
                    self._index = self._client.Index(name=self._index_name)
            else:
                # Pod-based index - connect by name
                logger.info(f"Connecting to pod-based index '{self._index_name}'")
                self._index = self._client.Index(name=self._index_name)

            # Verify index is accessible by getting stats
            try:
                stats = self._index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                dimension = stats.get('dimension', 0)
                
                logger.info(
                    f"Successfully connected to Pinecone index '{self._index_name}' "
                    f"(Total vectors: {total_vectors}, Dimension: {dimension})"
                )
                
                # Verify dimension matches expected (768 for text-embedding-005)
                if dimension and dimension != 768:
                    logger.warning(
                        f"Index dimension ({dimension}) doesn't match expected (768). "
                        "This may cause issues with embeddings."
                    )
                    
            except Exception as stats_error:
                logger.error(
                    f"Connected to index but failed to get stats: {str(stats_error)}. "
                    "Index may not be fully ready."
                )
                # Don't raise - connection might still work for operations

            self._initialized = True
            logger.info("Pinecone service initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize Pinecone service: {str(e)}",
                exc_info=True  # Include full traceback
            )
            # Reset state on failure
            self._initialized = False
            self._index = None
            self._client = None
            raise

    @property
    def index(self):
        """Get Pinecone index."""
        if not self._index:
            raise RuntimeError(
                "Pinecone index not initialized. Call initialize() first."
            )
        return self._index

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    async def upsert_vectors(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> int:
        """
        Upsert chunk embeddings to Pinecone.

        Args:
            chunks: List of Chunk instances
            embeddings: List of embedding vectors (one per chunk)

        Returns:
            Number of vectors upserted

        Raises:
            RuntimeError: If service is not initialized
            ValueError: If chunks and embeddings length mismatch
        """
        if not self._initialized:
            raise RuntimeError(
                "Pinecone service not initialized. Call initialize() first."
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "length mismatch"
            )

        if not embeddings:
            logger.warning("No embeddings provided, skipping Pinecone upsert")
            return 0

        # Convert embeddings to lists if they're TextEmbedding objects
        # Vertex AI returns TextEmbeddingResponse objects that need to be converted
        converted_embeddings = []
        for i, emb in enumerate(embeddings):
            try:
                if hasattr(emb, 'values'):
                    # TextEmbeddingResponse object - extract .values
                    emb_values = emb.values
                    if isinstance(emb_values, list):
                        # Ensure it's a list of numbers
                        if len(emb_values) > 0 and isinstance(emb_values[0], (int, float)):
                            converted_embeddings.append(emb_values)
                        else:
                            converted_embeddings.append([float(v) for v in emb_values])
                    else:
                        converted_embeddings.append(list(emb_values))
                elif isinstance(emb, list):
                    # Already a list - verify it's numbers
                    if len(emb) > 0 and isinstance(emb[0], (int, float)):
                        converted_embeddings.append(emb)
                    else:
                        converted_embeddings.append([float(v) for v in emb])
                elif hasattr(emb, '__iter__') and not isinstance(emb, str):
                    # Iterable - convert to list
                    converted_embeddings.append(list(emb))
                else:
                    logger.warning(f"Unexpected embedding type at index {i}: {type(emb)}")
                    converted_embeddings.append(list(emb))
            except Exception as e:
                logger.error(
                    f"Failed to convert embedding at index {i} (type: {type(emb)}): {str(e)}"
                )
                raise ValueError(f"Cannot convert embedding to list: {type(emb)}") from e
        
        embeddings = converted_embeddings

        # Verify embedding dimension matches index
        embedding_dim = len(embeddings[0])
        if embedding_dim == 0:
            raise ValueError("Embedding dimension is 0")

        try:
            # Prepare vectors for upsert (new SDK format)
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings):
                # Use chunk's embedding_id if available, otherwise use chunk id
                vector_id = chunk.embedding_id or chunk.id

                # Get metadata from chunk
                metadata = chunk.model_dump_for_pinecone()["metadata"]

                # New SDK format: (id, values, metadata) tuple
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                })

            # Upsert in batches (Pinecone recommends batches of 100)
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i : i + batch_size]
                self._index.upsert(vectors=batch)
                total_upserted += len(batch)
                logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")

            logger.info(
                f"Upserted {total_upserted} vectors to Pinecone index '{self._index_name}'"
            )

            return total_upserted

        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {str(e)}")
            raise

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> List[Dict]:
        """
        Search for similar vectors in Pinecone.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (e.g., {"document_id": "doc_123"})
            include_metadata: Whether to include metadata in results

        Returns:
            List of search results with metadata and scores

        Raises:
            RuntimeError: If service is not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "Pinecone service not initialized. Call initialize() first."
            )

        try:
            # Convert query_vector to list if it's a TextEmbedding object
            if hasattr(query_vector, 'values'):
                # TextEmbeddingResponse object - extract .values
                emb_values = query_vector.values
                if isinstance(emb_values, list):
                    # Ensure it's a list of numbers
                    if len(emb_values) > 0 and isinstance(emb_values[0], (int, float)):
                        query_vector = emb_values
                    else:
                        query_vector = [float(v) for v in emb_values]
                else:
                    query_vector = list(emb_values)
            elif not isinstance(query_vector, list):
                # Try to convert to list
                if hasattr(query_vector, '__iter__') and not isinstance(query_vector, str):
                    query_vector = list(query_vector)
                else:
                    raise ValueError(
                        f"Query vector must be a list, got {type(query_vector)}. "
                        f"If it's a TextEmbedding object, it should have a .values attribute."
                    )
            
            # Final validation - ensure it's a list of numbers
            if not isinstance(query_vector, list):
                raise ValueError(f"Query vector must be a list, got {type(query_vector)}")
            if len(query_vector) == 0:
                raise ValueError("Query vector is empty")
            if not isinstance(query_vector[0], (int, float)):
                # Try to convert to floats
                query_vector = [float(v) for v in query_vector]
            
            # Perform search (new SDK API)
            query_response = self._index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter,  # Filter support depends on Pinecone plan
            )

            # Format results (new SDK format)
            results = []
            if hasattr(query_response, 'matches'):
                for match in query_response.matches:
                    results.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": getattr(match, 'metadata', {}) or {},
                    })
            elif isinstance(query_response, dict):
                # Handle dict response format
                for match in query_response.get('matches', []):
                    results.append({
                        "id": match.get('id'),
                        "score": match.get('score', 0.0),
                        "metadata": match.get('metadata', {}),
                    })

            logger.info(
                f"Pinecone search returned {len(results)} results "
                f"(top_k={top_k}, filter={filter})"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to search Pinecone: {str(e)}")
            raise

    async def delete_vectors(self, document_id: str) -> int:
        """
        Delete all vectors for a document from Pinecone.

        Args:
            document_id: Document ID to delete vectors for

        Returns:
            Number of vectors deleted

        Raises:
            RuntimeError: If service is not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "Pinecone service not initialized. Call initialize() first."
            )

        try:
            # Delete by metadata filter
            # Note: This requires specific Pinecone plan features
            logger.warning(
                f"Delete by filter not fully implemented. "
                f"Consider tracking vector IDs for document {document_id}"
            )

            # For proper implementation, you'd need to:
            # 1. Query for all vectors with document_id filter
            # 2. Extract their IDs
            # 3. Delete by IDs using: self._index.delete(ids=[...])
            # Or use delete_by_filter if available in your Pinecone plan

            return 0

        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {str(e)}")
            raise

    def get_index_stats(self) -> Dict:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics

        Raises:
            RuntimeError: If service is not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "Pinecone service not initialized. Call initialize() first."
            )

        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone index stats: {str(e)}")
            raise

    def close(self):
        """Close Pinecone client connection."""
        if self._index:
            self._index = None
        if self._client:
            self._client = None
        self._initialized = False
        logger.info("Pinecone service closed")


# Global service instance
pinecone_service = PineconeService()
