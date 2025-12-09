"""
Firestore service for database operations.

This module provides async methods for CRUD operations on Firestore collections:
- Documents
- Chunks
- Analyses
- Users

Includes batch operations and transaction support for critical operations.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from google.cloud import firestore
from google.cloud.firestore_v1 import SERVER_TIMESTAMP
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.async_transaction import AsyncTransaction
from google.cloud.firestore_v1.base_query import BaseQuery

from config import settings
from models.analysis import Analysis
from models.chunk import Chunk
from models.document import Document
from models.user import User

logger = logging.getLogger(__name__)


class FirestoreService:
    """Async Firestore service for database operations."""

    def __init__(self):
        """Initialize Firestore client."""
        self._client: Optional[AsyncClient] = None
        self._db: Optional[AsyncClient] = None

    async def initialize(self):
        """Initialize Firestore client and database connection."""
        if self._client is not None:
            # Already initialized, skip
            return
        
        try:
            self._client = AsyncClient(
                project=settings.gcp_project_id,
                database=settings.firestore_database_id,
            )
            self._db = self._client
            logger.info(
                f"Firestore client initialized for project: {settings.gcp_project_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {str(e)}")
            raise

    async def close(self):
        """Close Firestore client connection."""
        if self._client:
            await self._client.close()
            logger.info("Firestore client closed")

    @asynccontextmanager
    async def transaction(self):
        """Async context manager for Firestore transactions."""
        if not self._db:
            await self.initialize()

        transaction = self._db.transaction()
        try:
            yield transaction
            await transaction.commit()
        except Exception as e:
            await transaction.rollback()
            raise

    @property
    def db(self) -> AsyncClient:
        """Get Firestore database instance."""
        if not self._db:
            raise RuntimeError("Firestore client not initialized. Call initialize() first.")
        return self._db

    # ============================================================================
    # Document Operations
    # ============================================================================

    async def create_document(self, document: Document) -> str:
        """
        Create a new document record in Firestore.

        This method stores a document's metadata in the Firestore 'documents' collection.
        The document is serialized using model_dump_for_firestore() to handle datetime
        objects and other Firestore-incompatible types.

        Args:
            document: Document model instance with all required fields populated.
                Must include: id, user_id, filename, storage_path, status, metadata

        Returns:
            str: The document ID (same as document.id)

        Raises:
            Exception: If Firestore write operation fails. The error is logged
                with context before re-raising.

        Example:
            ```python
            document = Document(
                id="doc_123",
                user_id="user_456",
                filename="zoning_map.pdf",
                storage_path="gs://bucket/documents/doc_123.pdf",
                status=DocumentStatus.UPLOADED,
                metadata=DocumentMetadata(...)
            )
            doc_id = await firestore_service.create_document(document)
            # Returns: "doc_123"
            ```

        Note:
            - Document ID must be unique within the collection
            - Timestamps are automatically converted to ISO format strings
            - This operation is not transactional (use transaction context manager for atomic operations)
        """
        try:
            doc_ref = self.db.collection("documents").document(document.id)
            await doc_ref.set(document.model_dump_for_firestore())
            logger.info(f"Created document: {document.id}")
            return document.id
        except Exception as e:
            logger.error(f"Failed to create document {document.id}: {str(e)}")
            raise

    async def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document instance or None if not found
        """
        try:
            doc_ref = self.db.collection("documents").document(document_id)
            doc_snapshot = await doc_ref.get()

            if not doc_snapshot.exists:
                logger.warning(f"Document not found: {document_id}")
                return None

            data = doc_snapshot.to_dict()
            return Document.from_firestore(document_id, data)
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            raise

    async def update_document(
        self, document_id: str, updates: dict, transaction: Optional[AsyncTransaction] = None
    ) -> bool:
        """
        Update a document.

        Args:
            document_id: Document ID
            updates: Dictionary of fields to update
            transaction: Optional transaction for atomic updates

        Returns:
            True if successful
        """
        try:
            doc_ref = self.db.collection("documents").document(document_id)

            # Add updated_at timestamp
            updates["updated_at"] = SERVER_TIMESTAMP

            if transaction:
                transaction.update(doc_ref, updates)
            else:
                await doc_ref.update(updates)

            logger.info(f"Updated document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {str(e)}")
            raise

    async def update_document_status(
        self, document_id: str, status: str, error_message: Optional[str] = None
    ) -> bool:
        """
        Update document status.

        Args:
            document_id: Document ID
            status: New status
            error_message: Optional error message

        Returns:
            True if successful
        """
        updates = {"status": status}
        if error_message:
            updates["error_message"] = error_message

        return await self.update_document(document_id, updates)

    async def list_documents(
        self, user_id: Optional[str] = None, limit: int = 50
    ) -> List[Document]:
        """
        List documents, optionally filtered by user.

        Args:
            user_id: Optional user ID filter
            limit: Maximum number of documents to return

        Returns:
            List of Document instances
        """
        try:
            query: BaseQuery = self.db.collection("documents")

            if user_id:
                query = query.where("user_id", "==", user_id)

            query = query.order_by("created_at", direction=firestore.Query.DESCENDING)
            query = query.limit(limit)

            docs = query.stream()
            documents = []

            async for doc in docs:
                data = doc.to_dict()
                documents.append(Document.from_firestore(doc.id, data))

            logger.info(f"Listed {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.

        Args:
            document_id: Document ID

        Returns:
            True if successful
        """
        try:
            doc_ref = self.db.collection("documents").document(document_id)
            await doc_ref.delete()
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise

    # ============================================================================
    # Chunk Operations
    # ============================================================================

    async def create_chunk(self, chunk: Chunk) -> str:
        """
        Create a new chunk in Firestore.

        Args:
            chunk: Chunk model instance

        Returns:
            Chunk ID
        """
        try:
            doc_ref = (
                self.db.collection("documents")
                .document(chunk.document_id)
                .collection("chunks")
                .document(chunk.id)
            )
            await doc_ref.set(chunk.model_dump_for_firestore())
            logger.debug(f"Created chunk: {chunk.id} for document: {chunk.document_id}")
            return chunk.id
        except Exception as e:
            logger.error(f"Failed to create chunk {chunk.id}: {str(e)}")
            raise

    async def create_chunks_batch(self, chunks: List[Chunk]) -> List[str]:
        """
        Create multiple chunks in a batch operation.

        Args:
            chunks: List of Chunk model instances

        Returns:
            List of created chunk IDs
        """
        try:
            batch = self.db.batch()
            chunk_ids = []

            for chunk in chunks:
                doc_ref = (
                    self.db.collection("documents")
                    .document(chunk.document_id)
                    .collection("chunks")
                    .document(chunk.id)
                )
                batch.set(doc_ref, chunk.model_dump_for_firestore())
                chunk_ids.append(chunk.id)

            await batch.commit()
            logger.info(f"Created {len(chunks)} chunks in batch")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to create chunks batch: {str(e)}")
            raise

    async def get_chunk(self, document_id: str, chunk_id: str) -> Optional[Chunk]:
        """
        Get a chunk by ID.

        Args:
            document_id: Parent document ID
            chunk_id: Chunk ID

        Returns:
            Chunk instance or None if not found
        """
        try:
            doc_ref = (
                self.db.collection("documents")
                .document(document_id)
                .collection("chunks")
                .document(chunk_id)
            )
            chunk_snapshot = await doc_ref.get()

            if not chunk_snapshot.exists:
                logger.warning(f"Chunk not found: {chunk_id}")
                return None

            data = chunk_snapshot.to_dict()
            return Chunk.from_firestore(chunk_id, data)
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {str(e)}")
            raise

    async def get_chunk(self, chunk_id: str, document_id: Optional[str] = None) -> Optional[Chunk]:
        """
        Get a specific chunk by ID.

        Args:
            chunk_id: Chunk ID (format: {document_id}_chunk_{index} or just chunk ID)
            document_id: Optional document ID to speed up lookup

        Returns:
            Chunk instance if found, None otherwise
        """
        try:
            # Extract document_id from chunk_id if not provided
            if not document_id:
                # Chunk ID format is typically: {document_id}_chunk_{index}
                if "_chunk_" in chunk_id:
                    document_id = chunk_id.split("_chunk_")[0]
                else:
                    # Try to find chunk by searching (less efficient)
                    logger.warning(
                        f"Document ID not provided for chunk {chunk_id}. "
                        "This will require a full search."
                    )
                    # For now, return None - would need to implement search
                    return None

            doc_ref = (
                self.db.collection("documents")
                .document(document_id)
                .collection("chunks")
                .document(chunk_id)
            )

            doc = await doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                return Chunk.from_firestore(chunk_id, data)
            else:
                logger.debug(f"Chunk not found: {chunk_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {str(e)}")
            return None

    async def list_chunks(
        self, document_id: str, limit: Optional[int] = None
    ) -> List[Chunk]:
        """
        List all chunks for a document.

        Args:
            document_id: Parent document ID
            limit: Optional limit on number of chunks

        Returns:
            List of Chunk instances
        """
        try:
            query = (
                self.db.collection("documents")
                .document(document_id)
                .collection("chunks")
                .order_by("metadata.chunk_index")
            )

            if limit:
                query = query.limit(limit)

            chunks = query.stream()
            chunk_list = []

            async for chunk in chunks:
                data = chunk.to_dict()
                chunk_list.append(Chunk.from_firestore(chunk.id, data))

            logger.debug(f"Listed {len(chunk_list)} chunks for document: {document_id}")
            return chunk_list
        except Exception as e:
            logger.error(f"Failed to list chunks for document {document_id}: {str(e)}")
            raise

    async def delete_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Parent document ID

        Returns:
            Number of chunks deleted
        """
        try:
            chunks_ref = (
                self.db.collection("documents")
                .document(document_id)
                .collection("chunks")
            )

            chunks = chunks_ref.stream()
            batch = self.db.batch()
            count = 0

            async for chunk in chunks:
                batch.delete(chunks_ref.document(chunk.id))
                count += 1

            if count > 0:
                await batch.commit()

            logger.info(f"Deleted {count} chunks for document: {document_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {str(e)}")
            raise

    # ============================================================================
    # Analysis Operations
    # ============================================================================

    async def create_analysis(self, analysis: Analysis) -> str:
        """
        Create a new analysis in Firestore.

        Args:
            analysis: Analysis model instance

        Returns:
            Analysis ID
        """
        try:
            doc_ref = (
                self.db.collection("documents")
                .document(analysis.document_id)
                .collection("analyses")
                .document(analysis.id)
            )
            await doc_ref.set(analysis.model_dump_for_firestore())
            logger.info(f"Created analysis: {analysis.id} for document: {analysis.document_id}")
            return analysis.id
        except Exception as e:
            logger.error(f"Failed to create analysis {analysis.id}: {str(e)}")
            raise

    async def get_analysis(self, document_id: str, analysis_id: str) -> Optional[Analysis]:
        """
        Get an analysis by ID.

        Args:
            document_id: Parent document ID
            analysis_id: Analysis ID

        Returns:
            Analysis instance or None if not found
        """
        try:
            doc_ref = (
                self.db.collection("documents")
                .document(document_id)
                .collection("analyses")
                .document(analysis_id)
            )
            analysis_snapshot = await doc_ref.get()

            if not analysis_snapshot.exists:
                logger.warning(f"Analysis not found: {analysis_id}")
                return None

            data = analysis_snapshot.to_dict()
            return Analysis.from_firestore(analysis_id, data)
        except Exception as e:
            logger.error(f"Failed to get analysis {analysis_id}: {str(e)}")
            raise

    async def get_latest_analysis(self, document_id: str) -> Optional[Analysis]:
        """
        Get the latest analysis for a document.

        Args:
            document_id: Parent document ID

        Returns:
            Latest Analysis instance or None if not found
        """
        try:
            query = (
                self.db.collection("documents")
                .document(document_id)
                .collection("analyses")
                .order_by("created_at", direction=firestore.Query.DESCENDING)
                .limit(1)
            )

            analyses = query.stream()
            async for analysis in analyses:
                data = analysis.to_dict()
                return Analysis.from_firestore(analysis.id, data)

            return None
        except Exception as e:
            logger.error(f"Failed to get latest analysis for document {document_id}: {str(e)}")
            raise

    async def update_analysis(
        self,
        document_id: str,
        analysis_id: str,
        updates: dict,
        transaction: Optional[AsyncTransaction] = None,
    ) -> bool:
        """
        Update an analysis.

        Args:
            document_id: Parent document ID
            analysis_id: Analysis ID
            updates: Dictionary of fields to update
            transaction: Optional transaction for atomic updates

        Returns:
            True if successful
        """
        try:
            doc_ref = (
                self.db.collection("documents")
                .document(document_id)
                .collection("analyses")
                .document(analysis_id)
            )

            # Add updated_at timestamp
            updates["updated_at"] = SERVER_TIMESTAMP

            if transaction:
                transaction.update(doc_ref, updates)
            else:
                await doc_ref.update(updates)

            logger.info(f"Updated analysis: {analysis_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update analysis {analysis_id}: {str(e)}")
            raise

    # ============================================================================
    # User Operations
    # ============================================================================

    async def create_user(self, user: User) -> str:
        """
        Create a new user in Firestore.

        Args:
            user: User model instance

        Returns:
            User ID
        """
        try:
            doc_ref = self.db.collection("users").document(user.id)
            await doc_ref.set(user.model_dump_for_firestore())
            logger.info(f"Created user: {user.id}")
            return user.id
        except Exception as e:
            logger.error(f"Failed to create user {user.id}: {str(e)}")
            raise

    async def get_user(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID.

        Args:
            user_id: User ID

        Returns:
            User instance or None if not found
        """
        try:
            doc_ref = self.db.collection("users").document(user_id)
            user_snapshot = await doc_ref.get()

            if not user_snapshot.exists:
                logger.warning(f"User not found: {user_id}")
                return None

            data = user_snapshot.to_dict()
            return User.from_firestore(user_id, data)
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {str(e)}")
            raise

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email.

        Args:
            email: User email

        Returns:
            User instance or None if not found
        """
        try:
            query = self.db.collection("users").where("email", "==", email).limit(1)
            users = query.stream()

            async for user in users:
                data = user.to_dict()
                return User.from_firestore(user.id, data)

            return None
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {str(e)}")
            raise

    async def update_user(
        self, user_id: str, updates: dict, transaction: Optional[AsyncTransaction] = None
    ) -> bool:
        """
        Update a user.

        Args:
            user_id: User ID
            updates: Dictionary of fields to update
            transaction: Optional transaction for atomic updates

        Returns:
            True if successful
        """
        try:
            doc_ref = self.db.collection("users").document(user_id)

            # Add updated_at timestamp
            updates["updated_at"] = SERVER_TIMESTAMP

            if transaction:
                transaction.update(doc_ref, updates)
            else:
                await doc_ref.update(updates)

            logger.info(f"Updated user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {str(e)}")
            raise


# Global service instance
firestore_service = FirestoreService()
