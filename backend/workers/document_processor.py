"""
Document processing logic for the worker.

This module handles the complete document processing pipeline:
1. Download from Cloud Storage
2. PDF parsing
3. Chunking
4. Embedding generation
5. Vector storage (Pinecone)
6. Analysis generation trigger
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from models.chunk import Chunk
from models.document import Document, DocumentStatus
from services import (
    analysis_service,
    storage_service,
)
from services.embedding_service import embedding_service
from services.firestore_service import firestore_service
from utils import chunker
from utils.pdf_parser import ParsedDocument, get_pdf_parser

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processor for handling document processing pipeline."""

    def __init__(self):
        """Initialize document processor."""
        self._initialized = False

    async def initialize(self):
        """Initialize services."""
        try:
            storage_service.initialize()
            embedding_service.initialize()
            await firestore_service.initialize()
            
            # Initialize Pinecone service (optional - will fall back to Firestore if fails)
            from services.pinecone_service import pinecone_service
            try:
                # Check if already initialized (might have been initialized by analysis_service)
                if not pinecone_service.is_initialized:
                    pinecone_service.initialize()
                    logger.info("Pinecone service initialized for document processor")
                else:
                    logger.info("Pinecone service already initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Pinecone service: {str(e)}. "
                    "Document processing will continue without vector storage.",
                    exc_info=True  # Include full traceback for debugging
                )
            
            await analysis_service.initialize()

            self._initialized = True
            logger.info("Document processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize document processor: {str(e)}")
            raise

    async def download_document(self, document: Document) -> bytes:
        """
        Download document from Cloud Storage.

        Args:
            document: Document model instance

        Returns:
            PDF file content as bytes

        Raises:
            Exception: If download fails
        """
        try:
            logger.info(f"Downloading document {document.id} from Cloud Storage")

            # Extract storage path from GCS URL
            storage_path = document.storage_path
            if storage_path.startswith("gs://"):
                # Extract path after bucket name
                parts = storage_path.split("/", 3)
                if len(parts) >= 4:
                    storage_path = parts[3]
                else:
                    storage_path = storage_path.replace(
                        f"gs://{storage_service._bucket_name}/", ""
                    )

            # Download file
            pdf_bytes = storage_service.download_file(storage_path)

            logger.info(
                f"Downloaded document {document.id}: {len(pdf_bytes)} bytes"
            )
            return pdf_bytes

        except Exception as e:
            logger.error(f"Failed to download document {document.id}: {str(e)}")
            raise

    async def parse_document(self, pdf_bytes: bytes, document_id: str) -> ParsedDocument:
        """
        Parse PDF document.

        Args:
            pdf_bytes: PDF file content
            document_id: Document ID for logging

        Returns:
            ParsedDocument instance

        Raises:
            Exception: If parsing fails
        """
        try:
            logger.info(f"Parsing PDF for document {document_id}")

            # Parse PDF using configured parser
            pdf_parser = get_pdf_parser()
            parsed_doc = pdf_parser.parse(pdf_bytes)

            logger.info(
                f"Parsed document {document_id}: {len(parsed_doc.pages)} pages, "
                f"{len(parsed_doc.tables)} tables, {len(parsed_doc.images)} images, "
                f"{len(parsed_doc.text_blocks)} text blocks"
            )

            return parsed_doc

        except Exception as e:
            logger.error(f"Failed to parse document {document_id}: {str(e)}")
            raise

    async def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        document_id: str,
        document_type: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Chunk parsed document.

        Args:
            parsed_doc: ParsedDocument instance
            document_id: Document ID
            document_type: Optional document type

        Returns:
            List of Chunk instances

        Raises:
            Exception: If chunking fails
        """
        try:
            logger.info(f"Chunking document {document_id}")

            # Chunk document
            chunks = chunker.chunk_document(
                parsed_doc=parsed_doc,
                document_id=document_id,
                document_type=document_type,
            )

            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk document {document_id}: {str(e)}")
            raise

    async def generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of Chunk instances

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")

            # Extract chunk texts
            chunk_texts = [chunk.content for chunk in chunks]

            # Generate embeddings in batches
            embeddings = await embedding_service.generate_embeddings(
                texts=chunk_texts,
                use_cache=True,
            )

            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    async def store_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]], document_id: str
    ):
        """
        Store chunks in Firestore and Pinecone.

        Args:
            chunks: List of Chunk instances
            embeddings: List of embedding vectors
            document_id: Document ID

        Raises:
            Exception: If storage fails
        """
        try:
            logger.info(f"Storing {len(chunks)} chunks for document {document_id}")

            # Store chunks in Firestore
            await firestore_service.create_chunks_batch(chunks)

            # Store embeddings in Pinecone
            from services.pinecone_service import pinecone_service
            if pinecone_service.is_initialized:
                try:
                    await pinecone_service.upsert_vectors(chunks, embeddings)
                    logger.info(f"Stored {len(chunks)} embeddings in Pinecone for document {document_id}")
                except Exception as e:
                    logger.warning(
                        f"Failed to store embeddings in Pinecone for document {document_id}: {str(e)}. "
                        "Chunks are still stored in Firestore."
                    )
            else:
                logger.warning(
                    "Pinecone service not initialized. Embeddings not stored in vector database."
                )

            logger.info(f"Stored chunks for document {document_id}")

        except Exception as e:
            logger.error(f"Failed to store chunks for document {document_id}: {str(e)}")
            raise

    async def trigger_analysis(
        self, document_id: str, document_type: Optional[str] = None
    ):
        """
        Trigger analysis generation for document.

        Args:
            document_id: Document ID
            document_type: Optional document type

        Raises:
            Exception: If analysis trigger fails
        """
        try:
            logger.info(f"Triggering analysis generation for document {document_id}")

            # Start analysis generation asynchronously
            # This will run in the background
            import asyncio

            asyncio.create_task(
                analysis_service.generate_analysis(document_id, document_type or "other")
            )

            logger.info(f"Analysis generation triggered for document {document_id}")

        except Exception as e:
            logger.error(
                f"Failed to trigger analysis for document {document_id}: {str(e)}"
            )
            # Don't raise - analysis failure shouldn't fail document processing
            logger.warning("Continuing despite analysis trigger failure")

    async def process_document(
        self,
        document_id: str,
        message_data: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> bool:
        """
        Process a document through the complete pipeline.

        This method handles the full document processing workflow:
        1. Download from Cloud Storage
        2. Parse PDF
        3. Chunk document
        4. Generate embeddings
        5. Store chunks and embeddings
        6. Trigger analysis generation

        Args:
            document_id: Document ID to process
            message_data: Optional message data from Pub/Sub
            resume_from: Optional stage to resume from (for resumable processing)

        Returns:
            True if processing completed successfully

        Raises:
            Exception: If processing fails
        """
        if not self._initialized:
            await self.initialize()

        document = None
        try:
            # Get document from Firestore
            document = await firestore_service.get_document(document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Check if already processed
            if document.status == DocumentStatus.INDEXED:
                logger.info(f"Document {document_id} already indexed, skipping")
                return True

            # Determine document type
            document_type = (
                message_data.get("document_type")
                if message_data
                else document.metadata.document_type.value
                if document.metadata.document_type
                else "other"
            )

            # Step 1: Download document (if not resuming from later stage)
            if resume_from not in ["parse", "chunk", "embed", "store"]:
                await firestore_service.update_document_status(
                    document_id, DocumentStatus.PROCESSING.value
                )

                pdf_bytes = await self.download_document(document)
            else:
                pdf_bytes = None
                logger.info(f"Resuming from stage: {resume_from}")

            # Step 2: Parse PDF (if not resuming from later stage)
            if resume_from not in ["chunk", "embed", "store"]:
                await firestore_service.update_document_status(
                    document_id, DocumentStatus.PARSING.value
                )

                if pdf_bytes is None:
                    pdf_bytes = await self.download_document(document)

                parsed_doc = await self.parse_document(pdf_bytes, document_id)

                # Update document metadata with parsing statistics
                await firestore_service.update_document(
                    document_id,
                    {
                        "metadata.page_count": len(parsed_doc.pages),
                        "metadata.tables_count": len(parsed_doc.tables),
                        "metadata.images_count": len(parsed_doc.images),
                        "metadata.text_blocks_count": len(parsed_doc.text_blocks),
                    },
                )
            else:
                parsed_doc = None
                logger.info(f"Skipping parsing, resuming from: {resume_from}")

            # Step 3: Chunk document (if not resuming from later stage)
            if resume_from not in ["embed", "store"]:
                await firestore_service.update_document_status(
                    document_id, DocumentStatus.CHUNKING.value
                )

                if parsed_doc is None:
                    # Need to re-parse if resuming
                    if pdf_bytes is None:
                        pdf_bytes = await self.download_document(document)
                    parsed_doc = await self.parse_document(pdf_bytes, document_id)

                chunks = await self.chunk_document(parsed_doc, document_id, document_type)
            else:
                # Retrieve existing chunks if resuming
                chunks = await firestore_service.list_chunks(document_id)
                logger.info(f"Retrieved {len(chunks)} existing chunks")

            # Step 4: Generate embeddings (if not resuming from store stage)
            if resume_from != "store":
                await firestore_service.update_document_status(
                    document_id, DocumentStatus.INDEXING.value
                )

                embeddings = await self.generate_embeddings(chunks)
            else:
                # Embeddings should already be generated
                embeddings = []
                logger.info("Skipping embedding generation, resuming from store")

            # Step 5: Store chunks and embeddings
            if resume_from != "store":
                await self.store_chunks(chunks, embeddings, document_id)
            else:
                logger.info("Skipping chunk storage, already stored")

            # Step 6: Update status to indexed
            await firestore_service.update_document_status(
                document_id, DocumentStatus.INDEXED.value
            )

            # Update processing timestamps
            await firestore_service.update_document(
                document_id,
                {
                    "processing_completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Step 7: Trigger analysis generation (async, non-blocking)
            try:
                await self.trigger_analysis(document_id, document_type)
            except Exception as e:
                logger.warning(
                    f"Analysis trigger failed for {document_id}: {str(e)}. "
                    "Document processing still succeeded."
                )

            logger.info(f"Successfully processed document: {document_id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to process document {document_id}: {str(e)}", exc_info=True
            )

            # Update document status to failed
            try:
                await firestore_service.update_document_status(
                    document_id,
                    DocumentStatus.FAILED.value,
                    error_message=str(e),
                )
            except Exception as update_error:
                logger.error(
                    f"Failed to update document status: {str(update_error)}"
                )

            raise

    async def check_resumable(self, document_id: str) -> Optional[str]:
        """
        Check if document processing can be resumed from a specific stage.

        Args:
            document_id: Document ID

        Returns:
            Stage to resume from, or None if cannot resume
        """
        try:
            document = await firestore_service.get_document(document_id)
            if not document:
                return None

            # Check if chunks exist (can resume from embedding)
            chunks = await firestore_service.list_chunks(document_id)
            if chunks:
                # Check if embeddings exist (can resume from storage)
                # For now, assume we can resume from chunk stage
                return "embed"

            # Check document status
            if document.status == DocumentStatus.PARSING:
                return "parse"
            elif document.status == DocumentStatus.CHUNKING:
                return "chunk"
            elif document.status == DocumentStatus.INDEXING:
                return "embed"

            return None

        except Exception as e:
            logger.error(f"Failed to check resumable status: {str(e)}")
            return None


# Global processor instance
document_processor = DocumentProcessor()
