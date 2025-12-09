"""
Document API routes.

This module provides endpoints for document upload, retrieval, and status checking.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, Response
from google.cloud.exceptions import NotFound
from pydantic import BaseModel, Field

from models.document import Document, DocumentMetadata, DocumentStatus, DocumentType
from services import (
    firestore_service,
    pubsub_service,
    storage_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes
ALLOWED_CONTENT_TYPES = ["application/pdf"]


# ============================================================================
# Request/Response Models
# ============================================================================


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    document_id: str = Field(..., description="Unique document identifier")
    status: str = Field(..., description="Document status")
    message: str = Field(..., description="Status message")
    storage_path: str = Field(..., description="Cloud Storage path to document")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "status": "uploaded",
                "message": "Document uploaded successfully. Processing started.",
                "storage_path": "gs://bucket/documents/doc_123456.pdf",
            }
        }


class ChunkSummary(BaseModel):
    """Summary model for chunk in document response."""

    id: str
    chunk_index: int
    content_preview: str = Field(..., description="First 200 characters of content")
    page_number: int
    element_type: Optional[str] = None
    is_table: bool = False
    is_image: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_123456_0",
                "chunk_index": 0,
                "content_preview": "This is a sample chunk of text from the document...",
                "page_number": 1,
                "element_type": "text",
                "is_table": False,
                "is_image": False,
            }
        }


class DocumentResponse(BaseModel):
    """Response model for document retrieval."""

    id: str
    user_id: str
    filename: str
    storage_path: str
    status: str
    metadata: dict
    created_at: str
    updated_at: str
    signed_url: Optional[str] = None
    document_type: Optional[str] = Field(
        None, description="Document type (zoning, risk, permit, other)"
    )
    chunks: Optional[List[ChunkSummary]] = Field(
        None, description="Document chunks if available"
    )
    chunk_count: Optional[int] = Field(None, description="Total number of chunks")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_123456",
                "user_id": "user_789",
                "filename": "zoning_map.pdf",
                "storage_path": "gs://bucket/documents/doc_123456.pdf",
                "status": "complete",
                "metadata": {
                    "filename": "zoning_map.pdf",
                    "file_size": 1024000,
                    "mime_type": "application/pdf",
                },
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "signed_url": "https://storage.googleapis.com/...",
                "chunks": [],
                "chunk_count": 0,
            }
        }


class DocumentStatusResponse(BaseModel):
    """Response model for document status."""

    document_id: str
    status: str
    progress: Optional[dict] = None
    error_message: Optional[str] = None
    updated_at: str

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "status": "processing",
                "progress": {
                    "stage": "chunking",
                    "chunks_processed": 5,
                    "total_chunks": 10,
                },
                "updated_at": "2024-01-01T00:00:00Z",
            }
        }


# ============================================================================
# Helper Functions
# ============================================================================


def validate_file_type(content_type: str) -> bool:
    """
    Validate file type is PDF.

    Args:
        content_type: MIME type of the file

    Returns:
        True if valid PDF, False otherwise
    """
    return content_type in ALLOWED_CONTENT_TYPES


def validate_file_size(file_size: int) -> bool:
    """
    Validate file size is within limits.

    Args:
        file_size: File size in bytes

    Returns:
        True if within limits, False otherwise
    """
    return file_size > 0 and file_size <= MAX_FILE_SIZE


async def read_upload_file(file: UploadFile) -> bytes:
    """
    Read uploaded file content as bytes.

    Args:
        file: UploadFile instance

    Returns:
        File content as bytes

    Raises:
        HTTPException: If file reading fails
    """
    try:
        content = await file.read()
        return content
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}",
        )


# ============================================================================
# API Endpoints
# ============================================================================


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Document",
    description="""
    Upload a PDF document for processing and analysis.
    
    This endpoint accepts a PDF file and initiates asynchronous processing:
    - File validation (PDF format, size limits)
    - Upload to Cloud Storage
    - Document metadata creation in Firestore
    - Pub/Sub message publication for worker processing
    
    **Authentication**: Currently open (add authentication middleware in production)
    
    **File Requirements**:
    - Format: PDF only
    - Maximum size: 50MB
    - Content: Real estate documents (zoning maps, risk assessments, permits)
    
    **Processing Flow**:
    1. File uploaded → Cloud Storage
    2. Document record created → Firestore
    3. Processing task published → Pub/Sub
    4. Worker processes document → Parsing, chunking, embedding, indexing
    5. Analysis can be generated → Via analysis endpoint
    
    **Example Request**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/documents/upload" \\
      -F "file=@document.pdf" \\
      -F "user_id=user_123" \\
      -F "document_type=zoning"
    ```
    
    **Example Response**:
    ```json
    {
      "document_id": "doc_abc123",
      "status": "uploaded",
      "message": "Document uploaded successfully. Processing started.",
      "storage_path": "gs://bucket/documents/doc_abc123/document.pdf"
    }
    ```
    """,
    responses={
        202: {
            "description": "Document uploaded successfully, processing started",
            "model": DocumentUploadResponse,
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "doc_abc123",
                        "status": "uploaded",
                        "message": "Document uploaded successfully. Processing started.",
                        "storage_path": "gs://bucket/documents/doc_abc123/document.pdf"
                    }
                }
            }
        },
        400: {
            "description": "Invalid file or validation error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid file type. Only PDF files are allowed."
                    }
                }
            }
        },
        413: {
            "description": "File too large",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "File too large. Maximum size is 50MB. Received: 75.2MB"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "An unexpected error occurred during document upload"
                    }
                }
            }
        },
    },
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload (max 50MB)"),
    user_id: str = Form(..., description="User ID uploading the document"),
    document_type: Optional[str] = Form(
        None, 
        description="Document type classification: 'zoning', 'risk', 'permit', or 'other'"
    ),
):
    """
    Upload a document for processing.

    This endpoint:
    1. Validates the uploaded file (PDF only, max 50MB)
    2. Uploads file to Cloud Storage
    3. Creates document record in Firestore
    4. Publishes Pub/Sub message for async processing
    5. Returns 202 Accepted with document ID

    **Note**: The document processing happens asynchronously. Use the status endpoint
    to check processing progress.

    Args:
        file: Uploaded PDF file (multipart/form-data)
        user_id: User ID uploading the document (required for authorization)
        document_type: Optional document type classification. Valid values:
            - 'zoning': Zoning maps and regulations
            - 'risk': Risk assessment documents
            - 'permit': Permit applications and approvals
            - 'other': Other real estate documents

    Returns:
        DocumentUploadResponse containing:
            - document_id: Unique identifier for the document
            - status: Current processing status ('uploaded')
            - message: Human-readable status message
            - storage_path: GCS path to the uploaded file

    Raises:
        HTTPException 400: If file validation fails (wrong type, empty file)
        HTTPException 413: If file exceeds size limit (50MB)
        HTTPException 500: If upload or database operation fails

    Example:
        ```python
        import requests
        
        with open('document.pdf', 'rb') as f:
            response = requests.post(
                'http://localhost:8000/api/v1/documents/upload',
                files={'file': f},
                data={
                    'user_id': 'user_123',
                    'document_type': 'zoning'
                }
            )
        print(response.json())
        ```
    """
    # Request ID would come from middleware in actual request
    # For now, we'll generate one for logging
    request_id = str(uuid.uuid4())

    try:
        # Validate file type
        if not validate_file_type(file.content_type or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Only PDF files are allowed. Received: {file.content_type}",
            )

        # Read file content
        file_content = await read_upload_file(file)

        # Validate file size
        file_size = len(file_content)
        if not validate_file_size(file_size):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.0f}MB. Received: {file_size / (1024 * 1024):.2f}MB",
            )

        # Generate document ID
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Services are initialized at startup, but ensure they're ready
        if not storage_service._client:
            storage_service.initialize()
        if not firestore_service._client:
            await firestore_service.initialize()
        if not pubsub_service._publisher:
            pubsub_service.initialize()

        # Upload to Cloud Storage
        storage_path = f"documents/{document_id}/{file.filename}"
        try:
            gcs_path = storage_service.upload_file(
                file=file_content,
                path=storage_path,
                content_type="application/pdf",
                metadata={
                    "document_id": document_id,
                    "user_id": user_id,
                    "original_filename": file.filename or "unknown.pdf",
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            logger.info(f"File uploaded to Cloud Storage: {gcs_path}")
        except Exception as e:
            logger.error(f"Failed to upload file to Cloud Storage: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to storage: {str(e)}",
            )

        # Create document metadata
        # Validate and convert document_type
        parsed_document_type = None
        if document_type:
            try:
                # Strip whitespace and convert to lowercase for consistency
                document_type_clean = document_type.strip().lower()
                parsed_document_type = DocumentType(document_type_clean)
                logger.info(f"Document type set to: {parsed_document_type.value}")
            except ValueError as e:
                logger.warning(
                    f"Invalid document_type '{document_type}'. Valid values are: "
                    f"{[dt.value for dt in DocumentType]}. Setting to None."
                )
                parsed_document_type = None
        
        document_metadata = DocumentMetadata(
            filename=file.filename or "unknown.pdf",
            file_size=file_size,
            mime_type="application/pdf",
            document_type=parsed_document_type,
            upload_source="api",
        )

        # Create document record
        document = Document(
            id=document_id,
            user_id=user_id,
            filename=file.filename or "unknown.pdf",
            storage_path=gcs_path,
            status=DocumentStatus.UPLOADED,
            metadata=document_metadata,
        )

        try:
            await firestore_service.create_document(document)
            logger.info(f"Document record created: {document_id}")
        except Exception as e:
            logger.error(f"Failed to create document record: {str(e)}")
            # Try to clean up uploaded file
            try:
                storage_service.delete_file(storage_path)
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create document record: {str(e)}",
            )

        # Publish Pub/Sub message for processing
        try:
            message_id = pubsub_service.publish_document_task(
                document_id=document_id,
                metadata={
                    "user_id": user_id,
                    "storage_path": gcs_path,
                    "document_type": document_type or "other",
                    "filename": file.filename or "unknown.pdf",
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
                # Note: ordering_key removed - requires message ordering enabled on Pub/Sub topic
                # To enable ordering, configure the topic with message ordering enabled
                # ordering_key=user_id,  # Order by user for consistent processing
            )
            logger.info(f"Pub/Sub message published: {message_id}")
        except Exception as e:
            logger.error(f"Failed to publish Pub/Sub message: {str(e)}")
            # Document is already created, so we'll log the error but continue
            # The document can be manually reprocessed if needed
            logger.warning(
                f"Document {document_id} created but Pub/Sub message failed. "
                "Document may need manual processing."
            )

        # Return 202 Accepted
        return DocumentUploadResponse(
            document_id=document_id,
            status="uploaded",
            message="Document uploaded successfully. Processing started.",
            storage_path=gcs_path,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in document upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during document upload",
        )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document",
    description="Retrieve document metadata, signed URL for file access, and chunks if available",
    responses={
        200: {"description": "Document retrieved successfully"},
        403: {"description": "Forbidden - User does not have access to this document"},
        404: {"description": "Document not found"},
    },
    tags=["Documents"],
)
async def get_document(
    document_id: str,
    user_id: Optional[str] = None,
    include_chunks: bool = False,
):
    """
    Get document by ID.

    This endpoint:
    1. Retrieves document metadata from Firestore
    2. Generates signed URL for file access
    3. Optionally includes document chunks
    4. Performs authorization check if user_id is provided

    Args:
        document_id: Document ID
        user_id: Optional user ID for authorization check (query parameter)
        include_chunks: Whether to include chunks in response (default: False)

    Returns:
        DocumentResponse with document metadata, signed URL, and optional chunks

    Raises:
        HTTPException: If document not found or user unauthorized
    """
    try:
        # Services are initialized at startup, but ensure they're ready
        if not firestore_service._client:
            await firestore_service.initialize()
        if not storage_service._client:
            storage_service.initialize()

        # Get document from Firestore
        document = await firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Authorization check: Verify user owns the document
        if user_id and document.user_id != user_id:
            logger.warning(
                f"Unauthorized access attempt: user {user_id} tried to access document {document_id} "
                f"owned by {document.user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this document",
            )

        # Generate signed URL for file access (1 hour expiration)
        signed_url = None
        try:
            # Extract path from gs:// URL
            gcs_path = document.storage_path
            logger.debug(f"Extracting storage path from: {gcs_path}")
            
            if gcs_path.startswith("gs://"):
                # Format: gs://bucket-name/path/to/file
                # Remove gs:// prefix and split by first /
                path_without_prefix = gcs_path[5:]  # Remove "gs://"
                # Find first / to separate bucket from path
                if "/" in path_without_prefix:
                    # Split into bucket and path
                    bucket_and_path = path_without_prefix.split("/", 1)
                    storage_path = bucket_and_path[1] if len(bucket_and_path) > 1 else ""
                else:
                    # No path after bucket name
                    storage_path = ""
                    logger.warning(f"No path found after bucket in: {gcs_path}")
            else:
                # Already a storage path, use as-is
                storage_path = gcs_path

            logger.info(f"Extracted storage path: {storage_path} (from {gcs_path})")
            
            if not storage_path:
                logger.warning(f"Empty storage path extracted from: {gcs_path}. Cannot generate signed URL.")
            else:
                signed_url = storage_service.generate_signed_url(
                    path=storage_path,
                    expiration=3600,  # 1 hour
                    method="GET",
                )
                logger.info(f"Successfully generated signed URL for: {storage_path}")
        except NotFound as e:
            logger.error(f"File not found in storage for path extracted from {document.storage_path}: {str(e)}")
            # Continue without signed URL - file might have been deleted
        except Exception as e:
            logger.error(
                f"Failed to generate signed URL for {document.storage_path}: {str(e)}", 
                exc_info=True
            )
            # Continue without signed URL

        # Retrieve chunks if requested
        chunks = None
        chunk_count = 0
        if include_chunks:
            try:
                chunk_list = await firestore_service.list_chunks(document_id)
                chunk_count = len(chunk_list)

                # Convert chunks to summary format (preview only)
                chunks = []
                for chunk in chunk_list[:50]:  # Limit to first 50 chunks
                    content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    chunks.append(
                        ChunkSummary(
                            id=chunk.id,
                            chunk_index=chunk.metadata.chunk_index,
                            content_preview=content_preview,
                            page_number=chunk.metadata.spatial_metadata.page_number,
                            element_type=chunk.metadata.spatial_metadata.element_type,
                            is_table=chunk.metadata.spatial_metadata.is_table,
                            is_image=chunk.metadata.spatial_metadata.is_image,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to retrieve chunks: {str(e)}")
                # Continue without chunks
                chunks = []

        # Extract document_type from metadata for convenience
        doc_type = document.metadata.document_type.value if document.metadata.document_type else None
        
        return DocumentResponse(
            id=document.id,
            user_id=document.user_id,
            filename=document.filename,
            storage_path=document.storage_path,
            status=document.status.value,
            metadata=document.metadata.model_dump(),
            created_at=document.created_at.isoformat(),
            updated_at=document.updated_at.isoformat(),
            signed_url=signed_url,
            document_type=doc_type,
            chunks=chunks if include_chunks else None,
            chunk_count=chunk_count if include_chunks else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document",
        )


@router.get(
    "/{document_id}/download",
    summary="Download Document",
    description="Download the original document file",
    responses={
        200: {"description": "Document file downloaded successfully", "content": {"application/pdf": {}}},
        403: {"description": "Forbidden - User does not have access to this document"},
        404: {"description": "Document not found"},
    },
    tags=["Documents"],
)
async def download_document(
    document_id: str,
    user_id: Optional[str] = None,
):
    """
    Download the original document file.

    This endpoint:
    1. Retrieves document metadata from Firestore
    2. Downloads file from Cloud Storage
    3. Returns the file directly (not JSON)
    4. Performs authorization check if user_id is provided

    Args:
        document_id: Document ID
        user_id: Optional user ID for authorization check (query parameter)

    Returns:
        File content as binary response with appropriate Content-Type header

    Raises:
        HTTPException: If document not found, user unauthorized, or file download fails
    """
    try:
        # Services are initialized at startup, but ensure they're ready
        if not firestore_service._client:
            await firestore_service.initialize()
        if not storage_service._client:
            storage_service.initialize()

        # Get document from Firestore
        document = await firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Authorization check: Verify user owns the document
        if user_id and document.user_id != user_id:
            logger.warning(
                f"Unauthorized download attempt: user {user_id} tried to download document {document_id} "
                f"owned by {document.user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to download this document",
            )

        # Extract storage path from gs:// URL
        gcs_path = document.storage_path
        logger.debug(f"Extracting storage path from: {gcs_path}")
        
        if gcs_path.startswith("gs://"):
            # Format: gs://bucket-name/path/to/file
            # Remove gs:// prefix and split by first /
            path_without_prefix = gcs_path[5:]  # Remove "gs://"
            # Find first / to separate bucket from path
            if "/" in path_without_prefix:
                # Split into bucket and path
                bucket_and_path = path_without_prefix.split("/", 1)
                storage_path = bucket_and_path[1] if len(bucket_and_path) > 1 else ""
            else:
                # No path after bucket name
                storage_path = ""
                logger.warning(f"No path found after bucket in: {gcs_path}")
        else:
            # Already a storage path, use as-is
            storage_path = gcs_path

        logger.info(f"Extracted storage path: {storage_path} (from {gcs_path})")
        
        if not storage_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid storage path for document {document_id}",
            )

        # Download file from Cloud Storage
        try:
            file_content = storage_service.download_file(storage_path)
            logger.info(f"Downloaded file for document {document_id}: {len(file_content)} bytes")
        except NotFound:
            logger.error(f"File not found in storage for path: {storage_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found in storage",
            )
        except Exception as e:
            logger.error(f"Failed to download file from storage: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download document file: {str(e)}",
            )

        # Return file as binary response
        return Response(
            content=file_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{document.filename}"',
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download document",
        )


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get Document Status",
    description="Check the processing status of a document",
    responses={
        200: {"description": "Status retrieved successfully"},
        404: {"description": "Document not found"},
    },
    tags=["Documents"],
)
async def get_document_status(document_id: str):
    """
    Get document processing status.

    Args:
        document_id: Document ID

    Returns:
        DocumentStatusResponse with current status and progress

    Raises:
        HTTPException: If document not found
    """
    try:
        # Services are initialized at startup, but ensure they're ready
        if not firestore_service._client:
            await firestore_service.initialize()

        # Get document from Firestore
        document = await firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Build progress information based on status
        progress = None
        status_mapping = {
            DocumentStatus.UPLOADED: {"stage": "uploaded", "progress_percent": 10},
            DocumentStatus.PROCESSING: {"stage": "processing", "progress_percent": 20},
            DocumentStatus.PARSING: {"stage": "parsing", "progress_percent": 30},
            DocumentStatus.CHUNKING: {"stage": "chunking", "progress_percent": 50},
            DocumentStatus.INDEXING: {"stage": "indexing", "progress_percent": 70},
            DocumentStatus.INDEXED: {"stage": "indexed", "progress_percent": 80},
            DocumentStatus.ANALYZING: {"stage": "analyzing", "progress_percent": 90},
            DocumentStatus.COMPLETE: {"stage": "complete", "progress_percent": 100},
            DocumentStatus.FAILED: {"stage": "failed", "progress_percent": 0},
        }

        if document.status in status_mapping:
            progress = status_mapping[document.status].copy()
            
            # Add timing information if available
            if document.processing_started_at:
                progress["started_at"] = document.processing_started_at.isoformat()
            if document.processing_completed_at:
                progress["completed_at"] = document.processing_completed_at.isoformat()

        return DocumentStatusResponse(
            document_id=document.id,
            status=document.status.value,
            progress=progress,
            error_message=document.error_message,
            updated_at=document.updated_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document status",
        )


@router.get(
    "",
    response_model=List[DocumentResponse],
    summary="List Documents",
    description="List documents, optionally filtered by user ID",
    responses={
        200: {"description": "Documents retrieved successfully"},
    },
    tags=["Documents"],
)
async def list_documents(
    user_id: Optional[str] = None,
    limit: int = 50,
):
    """
    List documents, optionally filtered by user.

    Args:
        user_id: Optional user ID filter (query parameter)
        limit: Maximum number of documents to return (default: 50)

    Returns:
        List of DocumentResponse objects
    """
    try:
        # Services are initialized at startup, but ensure they're ready
        if not firestore_service._client:
            await firestore_service.initialize()

        # List documents from Firestore
        documents = await firestore_service.list_documents(
            user_id=user_id, limit=limit
        )

        # Convert to response format
        document_responses = []
        for doc in documents:
            # Extract document_type from metadata for convenience
            doc_type = doc.metadata.document_type.value if doc.metadata.document_type else None
            document_responses.append(
                DocumentResponse(
                    id=doc.id,
                    user_id=doc.user_id,
                    filename=doc.filename,
                    storage_path=doc.storage_path,
                    status=doc.status.value,
                    metadata=doc.metadata.model_dump(),
                    created_at=doc.created_at.isoformat(),
                    updated_at=doc.updated_at.isoformat(),
                    signed_url=None,  # Don't generate signed URLs for list view
                    document_type=doc_type,
                )
            )

        return document_responses

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents",
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
    description="Delete a document and its associated data",
    responses={
        204: {"description": "Document deleted successfully"},
        403: {"description": "Forbidden - User does not have access to this document"},
        404: {"description": "Document not found"},
    },
    tags=["Documents"],
)
async def delete_document(
    document_id: str,
    user_id: Optional[str] = None,
):
    """
    Delete a document.

    This endpoint:
    1. Verifies user authorization (if user_id provided)
    2. Deletes document from Firestore
    3. Optionally deletes file from Cloud Storage

    Args:
        document_id: Document ID
        user_id: Optional user ID for authorization check (query parameter)

    Returns:
        204 No Content on success

    Raises:
        HTTPException: If document not found or user unauthorized
    """
    try:
        # Services are initialized at startup, but ensure they're ready
        if not firestore_service._client:
            await firestore_service.initialize()
        if not storage_service._client:
            storage_service.initialize()

        # Get document to verify existence and ownership
        document = await firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Authorization check
        if user_id and document.user_id != user_id:
            logger.warning(
                f"Unauthorized delete attempt: user {user_id} tried to delete document {document_id} "
                f"owned by {document.user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to delete this document",
            )

        # Delete document from Firestore (this will cascade delete chunks if configured)
        await firestore_service.delete_document(document_id)

        # Optionally delete file from Cloud Storage
        try:
            # Extract path from gs:// URL
            gcs_path = document.storage_path
            if gcs_path.startswith("gs://"):
                parts = gcs_path.split("/", 3)
                if len(parts) >= 4:
                    storage_path = parts[3]
                else:
                    storage_path = gcs_path.replace(
                        f"gs://{storage_service._bucket_name}/", ""
                    )
            else:
                storage_path = gcs_path

            storage_service.delete_file(storage_path)
            logger.info(f"Deleted file from Cloud Storage: {storage_path}")
        except Exception as e:
            logger.warning(f"Failed to delete file from Cloud Storage: {str(e)}")
            # Continue even if storage deletion fails

        return None  # 204 No Content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )
