"""
Analysis service for generating property analysis using LangChain orchestration.

This service orchestrates multi-step reasoning chains to generate comprehensive
property analysis from document chunks using Pinecone retrieval and LangChain.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain.prompts import PromptTemplate

from models.analysis import Analysis, AnalysisResult, AnalysisStatus, PropertyAnalysis
from prompts.analysis_prompt import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_PROMPT_TEMPLATE,
)
from prompts.formatting_prompt import (
    JSON_FORMATTING_PROMPT,
    JSON_VALIDATION_USER_PROMPT_TEMPLATE,
)
from config import settings
from services.embedding_service import embedding_service
from services.firestore_service import firestore_service
from services.llm_service import llm_service

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for generating property analysis using LangChain orchestration.
    
    Attributes:
        pinecone_service: Optional Pinecone service instance for vector search
    
    This service coordinates the complete analysis pipeline:
    1. Retrieves relevant document chunks from Pinecone using semantic search
    2. Orchestrates multi-step reasoning chains using LangChain
    3. Generates comprehensive property analysis using Vertex AI (Gemini Pro)
    4. Validates and formats output as structured JSON
    5. Stores analysis results in Firestore
    
    The service uses advanced prompt engineering with:
    - System prompts for analysis logic and JSON schema compliance
    - Retrieval prompts for refining document relevance
    - Formatting prompts for JSON validation and structure
    
    Example:
        ```python
        service = AnalysisService()
        await service.initialize()
        
        analysis = await service.generate_analysis(
            document_id="doc_123",
            document_type="zoning"
        )
        ```
    """

    def __init__(self):
        """
        Initialize analysis service.
        
        Creates a new AnalysisService instance. The service must be initialized
        via initialize() before use to set up dependencies (LLM service, Firestore, etc.).
        """
        self.pinecone_service = None  # Will be imported when Pinecone service is created
        self._initialized = False

    async def initialize(self):
        """
        Initialize analysis service and dependencies.
        
        Sets up all required services for analysis generation:
        - Embedding service (for query vectorization)
        - LLM service (LangChain + Vertex AI)
        - Firestore service (for storing results)
        - Pinecone service (for semantic search - when available)
        
        Raises:
            Exception: If any service initialization fails. All errors are logged
                with context before re-raising.
        
        Note:
            This method is idempotent - safe to call multiple times.
        """
        try:
            # Initialize services
            embedding_service.initialize()
            llm_service.initialize()
            await firestore_service.initialize()

            # Initialize Pinecone service
            from services.pinecone_service import pinecone_service
            self.pinecone_service = pinecone_service
            try:
                # Check if already initialized (might have been initialized by document_processor)
                if not pinecone_service.is_initialized:
                    pinecone_service.initialize()
                    logger.info("Pinecone service initialized for analysis service")
                else:
                    logger.info("Pinecone service already initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Pinecone service: {str(e)}. "
                    "Analysis will fall back to Firestore retrieval.",
                    exc_info=True  # Include full traceback for debugging
                )
                self.pinecone_service = None

            self._initialized = True
            logger.info("Analysis service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize analysis service: {str(e)}")
            raise

    async def _retrieve_relevant_chunks(
        self, document_id: str, query: str, top_k: int = 20
    ) -> List[Dict]:
        """
        Retrieve relevant chunks from Pinecone using semantic search.

        Args:
            document_id: Document ID to search within
            query: Search query
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(query)

            # Use Pinecone service if available, otherwise fall back to Firestore
            if self.pinecone_service and self.pinecone_service.is_initialized:
                # Search Pinecone for similar chunks
                pinecone_results = await self.pinecone_service.search(
                    query_vector=query_embedding,
                    top_k=top_k,
                    filter={"document_id": document_id},
                    include_metadata=True,
                )

                # Get full chunk data from Firestore using chunk IDs
                chunk_ids = [result["metadata"].get("chunk_id") for result in pinecone_results]
                chunks = []
                for chunk_id in chunk_ids:
                    if chunk_id:
                        # Get chunk from Firestore (chunk_id format: {document_id}_chunk_{index})
                        chunk = await firestore_service.get_chunk(chunk_id, document_id)
                        if chunk:
                            chunks.append(chunk)

                # Sort by Pinecone score (if we stored scores)
                # For now, just use the order from Pinecone results
                logger.info(
                    f"Retrieved {len(chunks)} chunks from Pinecone for document {document_id}"
                )
            else:
                # Fall back to Firestore retrieval
                logger.info(
                    f"Pinecone not available, using Firestore retrieval for document {document_id}"
                )
                chunks = await firestore_service.list_chunks(document_id, limit=top_k)

            # Format chunks for analysis
            formatted_chunks = []
            for chunk in chunks:
                formatted_chunks.append({
                    "content": chunk.content,
                    "metadata": {
                        "chunk_index": chunk.metadata.chunk_index,
                        "page_number": chunk.metadata.spatial_metadata.page_number,
                        "element_type": chunk.metadata.spatial_metadata.element_type,
                        "is_table": chunk.metadata.spatial_metadata.is_table,
                        "is_image": chunk.metadata.spatial_metadata.is_image,
                    },
                })

            logger.info(f"Retrieved {len(formatted_chunks)} chunks for document {document_id}")
            return formatted_chunks

        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {str(e)}")
            raise

    def _create_retrieval_chain(self) -> PromptTemplate:
        """Create prompt template for chunk retrieval and filtering."""
        # Simple retrieval - chunks are already retrieved, this is for context
        return PromptTemplate(
            input_variables=["chunks"],
            template="""Review the following document chunks and identify the most relevant information for property analysis.

Document Chunks:
{chunks}

Extract key information relevant to:
- Property address and identification
- Zoning classifications and restrictions
- Risk assessments (flood, fire, environmental)
- Permit requirements
- Building restrictions and limitations

Provide a summary of the most relevant information.""",
        )

    async def _run_analysis_pipeline(
        self, document_type: str, chunk_count: str, chunk_content: str
    ) -> Dict[str, str]:
        """
        Run multi-step analysis pipeline.

        Args:
            document_type: Type of document
            chunk_count: Number of chunks
            chunk_content: Formatted chunk content

        Returns:
            Dictionary with analysis_json and validated_json
        """
        # Step 1: Analysis generation
        analysis_prompt = llm_service.create_prompt_template(
            template=ANALYSIS_SYSTEM_PROMPT + "\n\n" + ANALYSIS_USER_PROMPT_TEMPLATE,
            input_variables=["document_type", "chunk_count", "chunk_content"],
        )

        analysis_chain = llm_service.create_chain(
            prompt=analysis_prompt,
            temperature=0.2,
            max_tokens=4096,
            output_key="analysis_json",
        )

        # Run analysis chain
        analysis_inputs = {
            "document_type": document_type,
            "chunk_count": chunk_count,
            "chunk_content": chunk_content,
        }

        analysis_output = await llm_service.run_chain_async(
            chain=analysis_chain,
            inputs=analysis_inputs,
            use_fallback=True,
        )

        analysis_json = analysis_output.get("analysis_json", "")

        # Step 2: JSON validation and formatting
        formatting_prompt = llm_service.create_prompt_template(
            template=JSON_FORMATTING_PROMPT + "\n\n" + JSON_VALIDATION_USER_PROMPT_TEMPLATE,
            input_variables=["json_output"],
        )

        formatting_chain = llm_service.create_chain(
            prompt=formatting_prompt,
            temperature=0.0,  # Deterministic for validation
            max_tokens=4096,
            output_key="validated_json",
        )

        # Run formatting chain
        formatting_inputs = {"json_output": analysis_json}

        formatting_output = await llm_service.run_chain_async(
            chain=formatting_chain,
            inputs=formatting_inputs,
            use_fallback=True,
        )

        validated_json = formatting_output.get("validated_json", "")

        return {
            "analysis_json": analysis_json,
            "validated_json": validated_json,
        }

    def _format_chunks_for_analysis(self, chunks: List[Dict]) -> str:
        """
        Format chunks into a single text for analysis.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Formatted chunk content string
        """
        formatted_parts = []

        for idx, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")

            part = f"--- Chunk {idx + 1} ---\n"
            part += f"Page: {metadata.get('page_number', 'Unknown')}\n"
            part += f"Type: {metadata.get('element_type', 'text')}\n"

            if metadata.get("is_table"):
                part += "[TABLE CONTENT]\n"
            elif metadata.get("is_image"):
                part += "[IMAGE WITH CAPTION]\n"

            part += f"\n{content}\n\n"

            formatted_parts.append(part)

        return "\n".join(formatted_parts)

    def _parse_analysis_json(self, json_str: str) -> PropertyAnalysis:
        """
        Parse JSON string into PropertyAnalysis model.

        Args:
            json_str: JSON string from LLM

        Returns:
            PropertyAnalysis instance

        Raises:
            ValueError: If JSON is invalid or doesn't match schema
        """
        try:
            # Clean JSON string (remove markdown code blocks if present)
            json_str = json_str.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            # Parse JSON
            data = json.loads(json_str)

            # Validate and create PropertyAnalysis
            analysis = PropertyAnalysis(**data)

            return analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            logger.error(f"JSON string: {json_str[:500]}")
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to create PropertyAnalysis: {str(e)}")
            raise ValueError(f"Invalid analysis data: {str(e)}")

    async def generate_analysis(
        self, document_id: str, document_type: str
    ) -> AnalysisResult:
        """
        Generate property analysis for a document.

        Args:
            document_id: Document ID to analyze
            document_type: Type of document (zoning, risk, permit, etc.)

        Returns:
            AnalysisResult with generated analysis

        Raises:
            Exception: If analysis generation fails
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now(timezone.utc)

        try:
            # Get document from Firestore
            document = await firestore_service.get_document(document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Create analysis record
            analysis_id = f"analysis_{document_id}_{int(start_time.timestamp())}"
            analysis = Analysis(
                id=analysis_id,
                document_id=document_id,
                status=AnalysisStatus.PROCESSING,
                started_at=start_time,
            )

            await firestore_service.create_analysis(analysis)

            # Update document status
            await firestore_service.update_document_status(
                document_id, "analyzing"
            )

            # Step 1: Retrieve relevant chunks
            logger.info(f"Retrieving chunks for document {document_id}")
            query = f"property analysis {document_type} zoning risk permits"
            chunks = await self._retrieve_relevant_chunks(
                document_id, query, top_k=20
            )

            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")

            # Step 2: Format chunks for analysis
            chunk_content = self._format_chunks_for_analysis(chunks)
            chunk_count = len(chunks)

            # Step 3: Run analysis pipeline (multi-step chain)
            logger.info(f"Generating analysis for document {document_id}")

            try:
                chain_output = await self._run_analysis_pipeline(
                    document_type=document_type,
                    chunk_count=str(chunk_count),
                    chunk_content=chunk_content,
                )
            except Exception as e:
                logger.error(f"Analysis pipeline failed: {str(e)}")
                raise

            # Step 4: Parse and validate JSON
            validated_json = chain_output.get("validated_json", "")
            if not validated_json:
                # Fallback to analysis_json if validated_json is empty
                validated_json = chain_output.get("analysis_json", "")

            try:
                property_analysis = self._parse_analysis_json(validated_json)
            except ValueError as e:
                logger.error(f"Failed to parse analysis JSON: {str(e)}")
                # Try to extract JSON from the response
                import re
                json_match = re.search(r"\{.*\}", validated_json, re.DOTALL)
                if json_match:
                    property_analysis = self._parse_analysis_json(json_match.group())
                else:
                    raise

            # Step 5: Create AnalysisResult
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            analysis_result = AnalysisResult(
                analysis=property_analysis,
                source_documents=[document_id],
                confidence_score=0.9,  # Could be calculated based on chunk relevance
                processing_time_seconds=processing_time,
                llm_model=settings.vertex_ai_model_name,
                chunks_retrieved=chunk_count,
            )

            # Step 6: Update analysis record
            analysis.result = analysis_result
            analysis.status = AnalysisStatus.COMPLETE
            analysis.completed_at = end_time
            analysis.updated_at = end_time

            await firestore_service.update_analysis(
                document_id,
                analysis_id,
                {
                    "result": analysis_result.model_dump(),
                    "status": AnalysisStatus.COMPLETE,
                    "completed_at": end_time.isoformat(),
                    "updated_at": end_time.isoformat(),
                },
            )

            # Update document status
            await firestore_service.update_document_status(document_id, "complete")

            logger.info(
                f"Analysis completed for document {document_id} in {processing_time:.2f}s"
            )

            return analysis_result

        except Exception as e:
            logger.error(f"Analysis generation failed for document {document_id}: {str(e)}")

            # Update analysis status to failed
            try:
                if "analysis_id" in locals():
                    await firestore_service.update_analysis(
                        document_id,
                        analysis_id,
                        {
                            "status": AnalysisStatus.FAILED,
                            "error_message": str(e),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )

                await firestore_service.update_document_status(
                    document_id, "failed"
                )
            except Exception as update_error:
                logger.error(f"Failed to update analysis status: {str(update_error)}")

            raise

    async def get_analysis(self, document_id: str) -> Optional[Analysis]:
        """
        Get the latest analysis for a document.

        Args:
            document_id: Document ID

        Returns:
            Analysis instance or None if not found
        """
        try:
            return await firestore_service.get_latest_analysis(document_id)
        except Exception as e:
            logger.error(f"Failed to get analysis for document {document_id}: {str(e)}")
            raise


# Global service instance
analysis_service = AnalysisService()
