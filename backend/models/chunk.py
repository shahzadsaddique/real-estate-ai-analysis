"""
Chunk data models.

This module defines Pydantic models for document chunks including
spatial metadata for layout-aware chunking.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SpatialMetadata(BaseModel):
    """Spatial metadata for layout-aware chunking."""

    page_number: int = Field(..., description="Page number (1-indexed)", ge=1)
    bbox: Optional[List[float]] = Field(
        None,
        description="Bounding box [x0, y0, x1, y1] in normalized coordinates (0-1)",
        min_length=4,
        max_length=4,
    )
    element_type: Optional[str] = Field(
        None, description="Type of element (text, table, image, header, footer)"
    )
    column_index: Optional[int] = Field(None, description="Column index for multi-column layouts", ge=0)
    row_index: Optional[int] = Field(None, description="Row index for tables", ge=0)
    is_table: bool = Field(default=False, description="Whether this chunk contains a table")
    is_image: bool = Field(default=False, description="Whether this chunk contains an image")
    has_caption: bool = Field(default=False, description="Whether this chunk has a caption")
    preceding_element: Optional[str] = Field(
        None, description="Type of preceding element for context"
    )
    following_element: Optional[str] = Field(
        None, description="Type of following element for context"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "page_number": 1,
                "bbox": [0.1, 0.2, 0.9, 0.8],
                "element_type": "text",
                "column_index": 0,
                "is_table": False,
                "is_image": False,
                "has_caption": False,
            }
        }


class ChunkMetadata(BaseModel):
    """Chunk metadata model."""

    chunk_index: int = Field(..., description="Index of chunk in document", ge=0)
    total_chunks: int = Field(..., description="Total number of chunks in document", ge=0)
    char_count: int = Field(..., description="Character count in chunk", ge=0)
    word_count: int = Field(..., description="Word count in chunk", ge=0)
    token_count: Optional[int] = Field(None, description="Estimated token count", ge=0)
    spatial_metadata: SpatialMetadata = Field(..., description="Spatial layout metadata")
    overlap_start: Optional[int] = Field(
        None, description="Number of characters overlapping with previous chunk", ge=0
    )
    overlap_end: Optional[int] = Field(
        None, description="Number of characters overlapping with next chunk", ge=0
    )
    language: Optional[str] = Field(None, description="Detected language code (e.g., 'en')")
    has_structure: bool = Field(
        default=False, description="Whether chunk contains structured data (tables, lists)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_index": 0,
                "total_chunks": 10,
                "char_count": 1500,
                "word_count": 250,
                "token_count": 375,
                "spatial_metadata": {
                    "page_number": 1,
                    "element_type": "text",
                },
                "overlap_start": 0,
                "overlap_end": 100,
            }
        }


class Chunk(BaseModel):
    """Chunk model representing a document chunk."""

    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content", min_length=1)
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    embedding_id: Optional[str] = Field(
        None, description="Pinecone vector ID for this chunk"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(), description="Creation timestamp"
    )
    additional_metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional metadata key-value pairs"
    )

    def model_dump_for_firestore(self) -> dict:
        """Serialize model for Firestore (convert datetime to ISO strings)."""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        return data

    def model_dump_for_pinecone(self) -> dict:
        """Serialize model for Pinecone vector storage."""
        return {
            "id": self.embedding_id or self.id,
            "values": [],  # Embedding values will be added separately
            "metadata": {
                "chunk_id": self.id,
                "document_id": self.document_id,
                "content": self.content[:1000],  # Truncate for metadata
                "chunk_index": self.metadata.chunk_index,
                "page_number": self.metadata.spatial_metadata.page_number,
                "element_type": self.metadata.spatial_metadata.element_type or "",
                "is_table": self.metadata.spatial_metadata.is_table,
                "is_image": self.metadata.spatial_metadata.is_image,
                "char_count": self.metadata.char_count,
                "word_count": self.metadata.word_count,
            },
        }

    @classmethod
    def from_firestore(cls, chunk_id: str, data: dict) -> "Chunk":
        """Create Chunk instance from Firestore data."""
        # Parse datetime strings back to datetime objects
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        data["id"] = chunk_id
        return cls(**data)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_123456_0",
                "document_id": "doc_123456",
                "content": "This is a sample chunk of text from the document...",
                "metadata": {
                    "chunk_index": 0,
                    "total_chunks": 10,
                    "char_count": 1500,
                    "word_count": 250,
                    "spatial_metadata": {
                        "page_number": 1,
                        "element_type": "text",
                    },
                },
                "created_at": "2024-01-01T00:00:00Z",
            }
        }
