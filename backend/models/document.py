"""
Document data models.

This module defines Pydantic models for document entities including
status enums, metadata, and document records.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    """Document processing status enum."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    INDEXED = "indexed"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Document type enum."""

    ZONING = "zoning"
    RISK = "risk"
    PERMIT = "permit"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    """Document metadata model."""

    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes", gt=0)
    mime_type: str = Field(default="application/pdf", description="MIME type")
    page_count: Optional[int] = Field(None, description="Number of pages", ge=0)
    tables_count: Optional[int] = Field(None, description="Number of tables extracted", ge=0)
    images_count: Optional[int] = Field(None, description="Number of images extracted", ge=0)
    text_blocks_count: Optional[int] = Field(None, description="Number of text blocks extracted", ge=0)
    document_type: Optional[DocumentType] = Field(
        None, description="Type of document (zoning, risk, permit, etc.)"
    )
    upload_source: Optional[str] = Field(None, description="Source of upload")
    additional_data: Dict[str, str] = Field(
        default_factory=dict, description="Additional metadata key-value pairs"
    )

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """Validate MIME type is PDF."""
        if v and not v.startswith("application/pdf"):
            raise ValueError("Only PDF files are supported")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "zoning_map_12345.pdf",
                "file_size": 1024000,
                "mime_type": "application/pdf",
                "page_count": 25,
                "tables_count": 1,
                "images_count": 8,
                "text_blocks_count": 238,
                "document_type": "zoning",
                "upload_source": "web",
                "additional_data": {"property_id": "12345"},
            }
        }


class Document(BaseModel):
    """Document model representing a processed document."""

    id: str = Field(..., description="Unique document identifier")
    user_id: str = Field(..., description="User who uploaded the document")
    filename: str = Field(..., description="Original filename")
    storage_path: str = Field(..., description="Cloud Storage path to the document")
    status: DocumentStatus = Field(
        default=DocumentStatus.UPLOADED, description="Current processing status"
    )
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Last update timestamp"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_started_at: Optional[datetime] = Field(
        None, description="When processing started"
    )
    processing_completed_at: Optional[datetime] = Field(
        None, description="When processing completed"
    )

    def model_dump_for_firestore(self) -> dict:
        """Serialize model for Firestore (convert datetime to ISO strings)."""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.processing_started_at:
            data["processing_started_at"] = self.processing_started_at.isoformat()
        if self.processing_completed_at:
            data["processing_completed_at"] = self.processing_completed_at.isoformat()
        if isinstance(data.get("metadata"), dict):
            # Ensure metadata is serializable
            pass
        return data

    @classmethod
    def from_firestore(cls, doc_id: str, data: dict) -> "Document":
        """Create Document instance from Firestore data."""
        # Parse datetime strings back to datetime objects
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if "processing_started_at" in data and data["processing_started_at"]:
            if isinstance(data["processing_started_at"], str):
                data["processing_started_at"] = datetime.fromisoformat(
                    data["processing_started_at"]
                )
        if "processing_completed_at" in data and data["processing_completed_at"]:
            if isinstance(data["processing_completed_at"], str):
                data["processing_completed_at"] = datetime.fromisoformat(
                    data["processing_completed_at"]
                )

        data["id"] = doc_id
        return cls(**data)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_123456",
                "user_id": "user_789",
                "filename": "zoning_map_12345.pdf",
                "storage_path": "gs://bucket/documents/doc_123456.pdf",
                "status": "uploaded",
                "metadata": {
                    "filename": "zoning_map_12345.pdf",
                    "file_size": 1024000,
                    "mime_type": "application/pdf",
                    "page_count": 25,
                    "document_type": "zoning",
                },
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        }
