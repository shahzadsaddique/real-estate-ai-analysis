"""
Intelligent chunking strategy for document processing.

This module provides layout-aware chunking that preserves document structure,
handles tables, images, multi-column layouts, and implements overlap windows.
"""

import logging
from typing import List, Optional

from models.chunk import Chunk, ChunkMetadata, SpatialMetadata
from utils.pdf_parser import (
    ImageBlock,
    ParsedDocument,
    Table,
    TextBlock,
)

logger = logging.getLogger(__name__)


class IntelligentChunker:
    """Intelligent chunker with layout awareness."""

    def __init__(
        self,
        default_chunk_size: int = 1000,
        default_overlap: float = 0.2,
        min_chunk_size: int = 100,
    ):
        """
        Initialize chunker.

        Args:
            default_chunk_size: Default chunk size in characters
            default_overlap: Default overlap ratio (0.0-1.0)
            min_chunk_size: Minimum chunk size to avoid tiny chunks
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.min_chunk_size = min_chunk_size

    def _calculate_word_count(self, text: str) -> int:
        """Calculate word count in text."""
        return len(text.split())

    def _calculate_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4

    def _split_text_with_overlap(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[tuple[str, int, int]]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to split
            chunk_size: Target chunk size in characters
            overlap: Overlap size in characters

        Returns:
            List of tuples: (chunk_text, start_pos, end_pos)
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Try to break at sentence boundary
            if end < text_length:
                # Look for sentence endings within last 20% of chunk
                search_start = max(start, end - chunk_size // 5)
                sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]

                for ending in sentence_endings:
                    last_ending = text.rfind(ending, search_start, end)
                    if last_ending != -1:
                        end = last_ending + len(ending)
                        break

                # If no sentence ending, try paragraph break
                if end == start + chunk_size:
                    para_break = text.rfind("\n\n", search_start, end)
                    if para_break != -1:
                        end = para_break + 2

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start, end))

            # Move start position with overlap
            start = end - overlap
            if start < 0:
                start = 0

            # Avoid infinite loop
            if start >= end:
                start = end

        return chunks

    def chunk_table(self, table: Table, document_id: str, chunk_index: int) -> Chunk:
        """
        Create a single chunk from a table, preserving table structure.

        Args:
            table: Table object
            document_id: Document ID
            chunk_index: Current chunk index

        Returns:
            Chunk containing table data
        """
        # Format table as text
        table_lines = []
        table_lines.append(f"Table on page {table.page_number}:\n")

        # Add table data
        for row in table.data:
            row_text = " | ".join(str(cell) if cell else "" for cell in row)
            table_lines.append(row_text)

        table_text = "\n".join(table_lines)

        # Create spatial metadata
        spatial_metadata = SpatialMetadata(
            page_number=table.page_number,
            bbox=table.bbox.to_list(),
            element_type="table",
            is_table=True,
            row_index=0,
        )

        # Create chunk metadata
        char_count = len(table_text)
        word_count = self._calculate_word_count(table_text)
        token_count = self._calculate_token_count(table_text)

        chunk_metadata = ChunkMetadata(
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            char_count=char_count,
            word_count=word_count,
            token_count=token_count,
            spatial_metadata=spatial_metadata,
            has_structure=True,
        )

        # Create chunk
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        chunk = Chunk(
            id=chunk_id,
            document_id=document_id,
            content=table_text,
            metadata=chunk_metadata,
        )

        logger.debug(f"Created table chunk: {chunk_id} ({char_count} chars)")
        return chunk

    def chunk_text_block(
        self,
        block: TextBlock,
        document_id: str,
        chunk_index: int,
        chunk_size: int,
        overlap: float,
    ) -> List[Chunk]:
        """
        Chunk a text block, respecting its boundaries.

        Args:
            block: TextBlock object
            document_id: Document ID
            chunk_index: Starting chunk index
            chunk_size: Target chunk size in characters
            overlap: Overlap ratio (0.0-1.0)

        Returns:
            List of Chunks from this text block
        """
        chunks = []
        text = block.text

        if not text or len(text.strip()) < self.min_chunk_size:
            # Too small to chunk, return as single chunk
            spatial_metadata = SpatialMetadata(
                page_number=block.page_number,
                bbox=block.bbox.to_list(),
                element_type="text",
                column_index=block.column_index,
            )

            char_count = len(text)
            word_count = self._calculate_word_count(text)
            token_count = self._calculate_token_count(text)

            chunk_metadata = ChunkMetadata(
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                char_count=char_count,
                word_count=word_count,
                token_count=token_count,
                spatial_metadata=spatial_metadata,
            )

            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                content=text,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)
            return chunks

        # Calculate overlap in characters
        overlap_chars = int(chunk_size * overlap)

        # Split text with overlap
        text_chunks = self._split_text_with_overlap(text, chunk_size, overlap_chars)

        for idx, (chunk_text, start_pos, end_pos) in enumerate(text_chunks):
            # Calculate overlap for this chunk
            overlap_start = start_pos if idx > 0 else 0
            overlap_end = (
                len(text) - end_pos if idx < len(text_chunks) - 1 else 0
            )

            # Create spatial metadata
            spatial_metadata = SpatialMetadata(
                page_number=block.page_number,
                bbox=block.bbox.to_list(),
                element_type="text",
                column_index=block.column_index,
                is_header=block.is_header,
                is_footer=block.is_footer,
            )

            # Create chunk metadata
            char_count = len(chunk_text)
            word_count = self._calculate_word_count(chunk_text)
            token_count = self._calculate_token_count(chunk_text)

            chunk_metadata = ChunkMetadata(
                chunk_index=chunk_index + idx,
                total_chunks=0,  # Will be updated later
                char_count=char_count,
                word_count=word_count,
                token_count=token_count,
                spatial_metadata=spatial_metadata,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )

            chunk_id = f"{document_id}_chunk_{chunk_index + idx}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                metadata=chunk_metadata,
            )

            chunks.append(chunk)

        return chunks

    def chunk_image_with_caption(
        self,
        image: ImageBlock,
        document_id: str,
        chunk_index: int,
    ) -> Chunk:
        """
        Create a chunk from an image with its caption.

        Args:
            image: ImageBlock object
            document_id: Document ID
            chunk_index: Current chunk index

        Returns:
            Chunk containing image description and caption
        """
        # Build image description
        image_parts = [f"Image on page {image.page_number}"]

        if image.caption:
            image_parts.append(f"Caption: {image.caption}")

        if image.alt_text:
            image_parts.append(f"Description: {image.alt_text}")

        image_text = "\n".join(image_parts)

        # Create spatial metadata
        spatial_metadata = SpatialMetadata(
            page_number=image.page_number,
            bbox=image.bbox.to_list(),
            element_type="image",
            is_image=True,
            has_caption=bool(image.caption),
        )

        # Create chunk metadata
        char_count = len(image_text)
        word_count = self._calculate_word_count(image_text)
        token_count = self._calculate_token_count(image_text)

        chunk_metadata = ChunkMetadata(
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            char_count=char_count,
            word_count=word_count,
            token_count=token_count,
            spatial_metadata=spatial_metadata,
        )

        # Create chunk
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        chunk = Chunk(
            id=chunk_id,
            document_id=document_id,
            content=image_text,
            metadata=chunk_metadata,
        )

        logger.debug(f"Created image chunk: {chunk_id} ({char_count} chars)")
        return chunk

    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        document_id: str,
        document_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[float] = None,
    ) -> List[Chunk]:
        """
        Chunk a parsed document with layout awareness.

        Args:
            parsed_doc: ParsedDocument from PDF parser
            document_id: Document ID
            document_type: Optional document type (zoning, risk, permit)
            chunk_size: Optional chunk size override
            overlap: Optional overlap ratio override

        Returns:
            List of Chunks with proper metadata
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        all_chunks = []
        chunk_index = 0

        # Process pages in order
        for page in parsed_doc.pages:
            # Group content by column for multi-column layouts
            column_content = {}

            # Collect text blocks by column
            for text_block in page.text_blocks:
                col_idx = text_block.column_index or 0
                if col_idx not in column_content:
                    column_content[col_idx] = {"text_blocks": [], "tables": [], "images": []}
                column_content[col_idx]["text_blocks"].append(text_block)

            # Collect tables by column (approximate)
            for table in page.tables:
                # Estimate column based on table position
                col_idx = int(table.bbox.x0 * 10) // 5  # Rough column estimation
                if col_idx not in column_content:
                    column_content[col_idx] = {"text_blocks": [], "tables": [], "images": []}
                column_content[col_idx]["tables"].append(table)

            # Collect images by column
            for image in page.images:
                col_idx = int(image.bbox.x0 * 10) // 5  # Rough column estimation
                if col_idx not in column_content:
                    column_content[col_idx] = {"text_blocks": [], "tables": [], "images": []}
                column_content[col_idx]["images"].append(image)

            # Process each column separately to respect column boundaries
            sorted_columns = sorted(column_content.keys())

            for col_idx in sorted_columns:
                content = column_content[col_idx]

                # Process content in order: text blocks, tables, images
                # Sort text blocks by y-position (top to bottom)
                text_blocks = sorted(
                    content["text_blocks"],
                    key=lambda b: (b.bbox.y0, b.bbox.x0),
                )

                # Process text blocks
                for text_block in text_blocks:
                    text_chunks = self.chunk_text_block(
                        text_block, document_id, chunk_index, chunk_size, overlap
                    )
                    all_chunks.extend(text_chunks)
                    chunk_index += len(text_chunks)

                # Process tables (each table is a single chunk)
                for table in content["tables"]:
                    table_chunk = self.chunk_table(table, document_id, chunk_index)
                    all_chunks.append(table_chunk)
                    chunk_index += 1

                # Process images (each image with caption is a single chunk)
                for image in content["images"]:
                    image_chunk = self.chunk_image_with_caption(
                        image, document_id, chunk_index
                    )
                    all_chunks.append(image_chunk)
                    chunk_index += 1

        # Update total_chunks in all chunk metadata
        total_chunks = len(all_chunks)
        for chunk in all_chunks:
            chunk.metadata.total_chunks = total_chunks

        # Add document type to additional metadata if provided
        if document_type:
            for chunk in all_chunks:
                chunk.additional_metadata["document_type"] = document_type

        logger.info(
            f"Chunked document {document_id}: {total_chunks} chunks created "
            f"(chunk_size={chunk_size}, overlap={overlap})"
        )

        return all_chunks

    def chunk_text(
        self,
        text: str,
        document_id: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[float] = None,
    ) -> List[Chunk]:
        """
        Simple text chunking without layout awareness.

        Useful for plain text or when layout information is not available.

        Args:
            text: Text to chunk
            document_id: Document ID
            chunk_size: Optional chunk size override
            overlap: Optional overlap ratio override

        Returns:
            List of Chunks
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        overlap_chars = int(chunk_size * overlap)
        text_chunks = self._split_text_with_overlap(text, chunk_size, overlap_chars)

        chunks = []
        for idx, (chunk_text, start_pos, end_pos) in enumerate(text_chunks):
            overlap_start = start_pos if idx > 0 else 0
            overlap_end = len(text) - end_pos if idx < len(text_chunks) - 1 else 0

            spatial_metadata = SpatialMetadata(
                page_number=1,
                bbox=[0.0, 0.0, 1.0, 1.0],  # Default bbox
                element_type="text",
            )

            char_count = len(chunk_text)
            word_count = self._calculate_word_count(chunk_text)
            token_count = self._calculate_token_count(chunk_text)

            chunk_metadata = ChunkMetadata(
                chunk_index=idx,
                total_chunks=len(text_chunks),
                char_count=char_count,
                word_count=word_count,
                token_count=token_count,
                spatial_metadata=spatial_metadata,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )

            chunk_id = f"{document_id}_chunk_{idx}"
            chunk = Chunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                metadata=chunk_metadata,
            )

            chunks.append(chunk)

        return chunks


# Global chunker instance
chunker = IntelligentChunker()
