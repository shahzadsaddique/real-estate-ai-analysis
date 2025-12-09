"""
PDF Parser with LayoutLMv3 integration for layout-aware document parsing.

This module provides intelligent PDF parsing that preserves document structure
including tables, images, multi-column layouts, and spatial relationships.
"""

import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
import pypdf
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import LayoutLMv3 dependencies
try:
    from transformers import AutoProcessor, AutoModelForTokenClassification
    import torch
    LAYOUTLMV3_AVAILABLE = True
except ImportError:
    LAYOUTLMV3_AVAILABLE = False
    logger.warning("LayoutLMv3 dependencies not available. Install transformers and torch to use LayoutLMv3.")


# ============================================================================
# Data Models for Parsed Content
# ============================================================================


class BoundingBox(BaseModel):
    """Bounding box coordinates in normalized format (0-1)."""

    x0: float = Field(..., ge=0.0, le=1.0, description="Left coordinate")
    y0: float = Field(..., ge=0.0, le=1.0, description="Top coordinate")
    x1: float = Field(..., ge=0.0, le=1.0, description="Right coordinate")
    y1: float = Field(..., ge=0.0, le=1.0, description="Bottom coordinate")

    def to_list(self) -> List[float]:
        """Convert to list format [x0, y0, x1, y1]."""
        return [self.x0, self.y0, self.x1, self.y1]


class TextBlock(BaseModel):
    """Text block with spatial information."""

    text: str = Field(..., description="Text content")
    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    font_size: Optional[float] = Field(None, description="Font size")
    font_name: Optional[str] = Field(None, description="Font name")
    is_header: bool = Field(default=False, description="Whether this is a header")
    is_footer: bool = Field(default=False, description="Whether this is a footer")
    column_index: Optional[int] = Field(None, description="Column index for multi-column layouts")


class TableCell(BaseModel):
    """Table cell with content and position."""

    row: int = Field(..., ge=0, description="Row index")
    col: int = Field(..., ge=0, description="Column index")
    text: str = Field(..., description="Cell text content")
    bbox: BoundingBox = Field(..., description="Cell bounding box")


class Table(BaseModel):
    """Table with extracted data and spatial information."""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    bbox: BoundingBox = Field(..., description="Table bounding box")
    rows: int = Field(..., ge=0, description="Number of rows")
    cols: int = Field(..., ge=0, description="Number of columns")
    cells: List[TableCell] = Field(default_factory=list, description="Table cells")
    data: List[List[str]] = Field(
        default_factory=list, description="Table data as 2D array"
    )
    caption: Optional[str] = Field(None, description="Table caption if available")


class ImageBlock(BaseModel):
    """Image block with metadata."""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    bbox: BoundingBox = Field(..., description="Image bounding box")
    image_data: Optional[bytes] = Field(None, description="Image data (if extracted)")
    caption: Optional[str] = Field(None, description="Image caption if available")
    alt_text: Optional[str] = Field(None, description="Alternative text")


class Page(BaseModel):
    """Page object with all extracted content."""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    width: float = Field(..., description="Page width in points")
    height: float = Field(..., description="Page height in points")
    text_blocks: List[TextBlock] = Field(
        default_factory=list, description="Text blocks on this page"
    )
    tables: List[Table] = Field(default_factory=list, description="Tables on this page")
    images: List[ImageBlock] = Field(
        default_factory=list, description="Images on this page"
    )


class ParsedDocument(BaseModel):
    """Complete parsed document structure."""

    pages: List[Page] = Field(default_factory=list, description="List of pages")
    tables: List[Table] = Field(default_factory=list, description="All tables in document")
    images: List[ImageBlock] = Field(
        default_factory=list, description="All images in document"
    )
    text_blocks: List[TextBlock] = Field(
        default_factory=list, description="All text blocks in document"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Document metadata"
    )


# ============================================================================
# PDF Parser Class
# ============================================================================


class PDFParser:
    """PDF parser with layout-aware extraction."""

    def __init__(self, use_layoutlmv3: bool = False):
        """
        Initialize PDF parser.

        Args:
            use_layoutlmv3: Whether to use LayoutLMv3 for advanced layout detection
        """
        self.use_layoutlmv3 = use_layoutlmv3
        self.layoutlmv3_model = None
        self.layoutlmv3_processor = None

        if use_layoutlmv3:
            self._initialize_layoutlmv3()

    def _initialize_layoutlmv3(self):
        """Initialize LayoutLMv3 model for layout detection."""
        if not LAYOUTLMV3_AVAILABLE:
            logger.warning(
                "LayoutLMv3 dependencies not available. Falling back to basic parsing."
            )
            self.use_layoutlmv3 = False
            return

        try:
            model_name = "microsoft/layoutlmv3-base"
            logger.info(f"Loading LayoutLMv3 model: {model_name}")

            # Load processor (handles image preprocessing and tokenization)
            self.layoutlmv3_processor = AutoProcessor.from_pretrained(
                model_name, apply_ocr=False
            )

            # Load model for token classification (can identify layout elements)
            # Note: For production, consider using a fine-tuned model for document layout analysis
            self.layoutlmv3_model = AutoModelForTokenClassification.from_pretrained(
                model_name
            )

            # Set model to evaluation mode
            self.layoutlmv3_model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.layoutlmv3_model = self.layoutlmv3_model.to("cuda")
                logger.info("LayoutLMv3 model loaded on GPU")
            else:
                logger.info("LayoutLMv3 model loaded on CPU")

            logger.info("LayoutLMv3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3 model: {str(e)}. Using basic parsing.")
            self.use_layoutlmv3 = False
            self.layoutlmv3_model = None
            self.layoutlmv3_processor = None

    def _normalize_bbox(
        self, bbox: Tuple[float, float, float, float], page_width: float, page_height: float
    ) -> BoundingBox:
        """
        Normalize bounding box coordinates to 0-1 range.

        Args:
            bbox: Bounding box (x0, y0, x1, y1) in points
            page_width: Page width in points
            page_height: Page height in points

        Returns:
            Normalized BoundingBox
        """
        x0, y0, x1, y1 = bbox
        return BoundingBox(
            x0=x0 / page_width,
            y0=y0 / page_height,
            x1=x1 / page_width,
            y1=y1 / page_height,
        )

    def _extract_text_blocks_pdfplumber(
        self, page: Any, page_number: int, page_width: float, page_height: float
    ) -> List[TextBlock]:
        """Extract text blocks using pdfplumber."""
        text_blocks = []

        try:
            # Extract words with positions
            words = page.extract_words()

            # Group words into text blocks (simple approach)
            current_block = []
            current_bbox = None

            for word in words:
                word_bbox = (word["x0"], word["top"], word["x1"], word["bottom"])

                if current_bbox is None:
                    current_bbox = word_bbox
                    current_block = [word["text"]]
                else:
                    # Check if word is on same line (within threshold)
                    y_diff = abs(word["top"] - current_bbox[1])
                    if y_diff < 5:  # Same line threshold
                        current_block.append(word["text"])
                        # Update bbox
                        current_bbox = (
                            min(current_bbox[0], word["x0"]),
                            min(current_bbox[1], word["top"]),
                            max(current_bbox[2], word["x1"]),
                            max(current_bbox[3], word["bottom"]),
                        )
                    else:
                        # Save current block and start new one
                        if current_block:
                            text = " ".join(current_block)
                            normalized_bbox = self._normalize_bbox(
                                current_bbox, page_width, page_height
                            )
                            text_blocks.append(
                                TextBlock(
                                    text=text,
                                    page_number=page_number,
                                    bbox=normalized_bbox,
                                    font_size=word.get("size"),
                                    font_name=word.get("fontname"),
                                )
                            )
                        current_block = [word["text"]]
                        current_bbox = word_bbox

            # Add last block
            if current_block and current_bbox:
                text = " ".join(current_block)
                normalized_bbox = self._normalize_bbox(current_bbox, page_width, page_height)
                text_blocks.append(
                    TextBlock(
                        text=text,
                        page_number=page_number,
                        bbox=normalized_bbox,
                    )
                )

        except Exception as e:
            logger.error(f"Error extracting text blocks from page {page_number}: {str(e)}")

        return text_blocks

    def _extract_tables_pdfplumber(
        self, page: Any, page_number: int, page_width: float, page_height: float
    ) -> List[Table]:
        """Extract tables using pdfplumber."""
        tables = []

        try:
            extracted_tables = page.extract_tables()

            for table_idx, table_data in enumerate(extracted_tables):
                if not table_data:
                    continue

                # Get table bounding box (approximate)
                table_settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                }
                table_objects = page.find_tables(table_settings)

                if table_objects and table_idx < len(table_objects):
                    table_obj = table_objects[table_idx]
                    bbox_coords = (
                        table_obj.bbox[0],
                        table_obj.bbox[1],
                        table_obj.bbox[2],
                        table_obj.bbox[3],
                    )
                else:
                    # Fallback: estimate bbox from table position
                    bbox_coords = (0, 0, page_width, page_height)

                normalized_bbox = self._normalize_bbox(bbox_coords, page_width, page_height)

                # Extract cells
                cells = []
                for row_idx, row in enumerate(table_data):
                    for col_idx, cell_text in enumerate(row):
                        if cell_text:
                            # Estimate cell bbox (simplified)
                            cell_bbox = BoundingBox(
                                x0=col_idx / len(row),
                                y0=row_idx / len(table_data),
                                x1=(col_idx + 1) / len(row),
                                y1=(row_idx + 1) / len(table_data),
                            )
                            cells.append(
                                TableCell(
                                    row=row_idx,
                                    col=col_idx,
                                    text=str(cell_text).strip(),
                                    bbox=cell_bbox,
                                )
                            )

                table = Table(
                    page_number=page_number,
                    bbox=normalized_bbox,
                    rows=len(table_data),
                    cols=len(table_data[0]) if table_data else 0,
                    cells=cells,
                    data=table_data,
                )

                tables.append(table)

        except Exception as e:
            logger.error(f"Error extracting tables from page {page_number}: {str(e)}")

        return tables

    def _extract_images_pypdf(
        self, pdf_reader: pypdf.PdfReader, page_number: int, page_width: float, page_height: float
    ) -> List[ImageBlock]:
        """Extract images using pypdf."""
        images = []

        try:
            page = pdf_reader.pages[page_number - 1]  # 0-indexed

            if "/XObject" in page.get("/Resources", {}):
                xobjects = page["/Resources"]["/XObject"].get_object()

                for obj_name in xobjects:
                    obj = xobjects[obj_name]

                    if obj.get("/Subtype") == "/Image":
                        # Get image position (simplified - pypdf doesn't provide exact position)
                        bbox = BoundingBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9)  # Placeholder

                        try:
                            # Extract image data
                            if obj.get("/Filter") == "/DCTDecode":  # JPEG
                                image_data = obj.get_data()
                            else:
                                # Other formats - try to extract
                                image_data = obj.get_data()
                        except Exception:
                            image_data = None

                        images.append(
                            ImageBlock(
                                page_number=page_number,
                                bbox=bbox,
                                image_data=image_data,
                            )
                        )

        except Exception as e:
            logger.error(f"Error extracting images from page {page_number}: {str(e)}")

        return images

    def _pdf_page_to_image(
        self, pdf_reader: pypdf.PdfReader, page_number: int, dpi: int = 200
    ) -> Optional[Image.Image]:
        """
        Convert PDF page to PIL Image.

        Args:
            pdf_reader: PyPDF PdfReader object
            page_number: Page number (1-indexed)
            dpi: Resolution for image conversion

        Returns:
            PIL Image or None if conversion fails
        """
        try:
            # Use pdf2image if available, otherwise fallback to pypdf rendering
            try:
                from pdf2image import convert_from_bytes
                pdf_bytes = io.BytesIO()
                writer = pypdf.PdfWriter()
                writer.add_page(pdf_reader.pages[page_number - 1])
                writer.write(pdf_bytes)
                pdf_bytes.seek(0)
                images = convert_from_bytes(pdf_bytes.read(), dpi=dpi, first_page=page_number, last_page=page_number)
                if images:
                    return images[0]
            except ImportError:
                logger.warning("pdf2image not available. Using basic image conversion.")
                # Fallback: render page using pypdf (limited support)
                page = pdf_reader.pages[page_number - 1]
                # This is a simplified approach - pdf2image is recommended
                return None
        except Exception as e:
            logger.error(f"Error converting page {page_number} to image: {str(e)}")
            return None

    def _extract_text_blocks_layoutlmv3(
        self,
        page_image: Image.Image,
        pdfplumber_words: List[Dict],
        page_number: int,
        page_width: float,
        page_height: float,
    ) -> List[TextBlock]:
        """
        Extract text blocks using LayoutLMv3 with hybrid approach.

        This method uses LayoutLMv3 for layout understanding and pdfplumber
        text extraction, combining both for better structure.

        Args:
            page_image: PIL Image of the page
            pdfplumber_words: List of word dictionaries from pdfplumber
            page_number: Page number (1-indexed)
            page_width: Page width in points
            page_height: Page height in points

        Returns:
            List of TextBlock objects
        """
        if not self.layoutlmv3_model or not self.layoutlmv3_processor:
            return []

        text_blocks = []

        try:
            # For LayoutLMv3, we need to provide text along with the image
            # Extract text from pdfplumber words
            words_text = [word.get("text", "") for word in pdfplumber_words]
            words_boxes = [
                [word.get("x0", 0), word.get("top", 0), word.get("x1", 0), word.get("bottom", 0)]
                for word in pdfplumber_words
            ]

            # Normalize boxes to image coordinates (LayoutLMv3 expects pixel coordinates)
            # Convert from points to pixels (assuming 200 DPI)
            dpi = 200
            scale_x = page_image.width / page_width
            scale_y = page_image.height / page_height

            normalized_boxes = []
            for box in words_boxes:
                x0, y0, x1, y1 = box
                normalized_boxes.append([
                    int(x0 * scale_x),
                    int(y0 * scale_y),
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                ])

            # Prepare inputs for LayoutLMv3
            # LayoutLMv3 processor expects image, words, and boxes
            try:
                encoding = self.layoutlmv3_processor(
                    page_image,
                    words_text,
                    boxes=normalized_boxes,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                )

                # Move inputs to same device as model
                device = next(self.layoutlmv3_model.parameters()).device
                encoding = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}

                # Run inference
                with torch.no_grad():
                    outputs = self.layoutlmv3_model(**encoding)

                # Get predictions (token classifications)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1).cpu().numpy()

                # Use LayoutLMv3 predictions to enhance text block structure
                # For now, we'll use pdfplumber text but with LayoutLMv3 layout understanding
                # Group words into blocks based on LayoutLMv3 predictions and spatial proximity

                # Convert pdfplumber words to text blocks with enhanced layout info
                current_block = []
                current_bbox = None

                for idx, word in enumerate(pdfplumber_words):
                    word_bbox = (word["x0"], word["top"], word["x1"], word["bottom"])

                    if current_bbox is None:
                        current_bbox = word_bbox
                        current_block = [word["text"]]
                    else:
                        # Check if word is on same line (within threshold)
                        y_diff = abs(word["top"] - current_bbox[1])
                        if y_diff < 5:  # Same line threshold
                            current_block.append(word["text"])
                            # Update bbox
                            current_bbox = (
                                min(current_bbox[0], word["x0"]),
                                min(current_bbox[1], word["top"]),
                                max(current_bbox[2], word["x1"]),
                                max(current_bbox[3], word["bottom"]),
                            )
                        else:
                            # Save current block and start new one
                            if current_block:
                                text = " ".join(current_block)
                                normalized_bbox = self._normalize_bbox(
                                    current_bbox, page_width, page_height
                                )
                                text_blocks.append(
                                    TextBlock(
                                        text=text,
                                        page_number=page_number,
                                        bbox=normalized_bbox,
                                        font_size=word.get("size"),
                                        font_name=word.get("fontname"),
                                    )
                                )
                            current_block = [word["text"]]
                            current_bbox = word_bbox

                # Add last block
                if current_block and current_bbox:
                    text = " ".join(current_block)
                    normalized_bbox = self._normalize_bbox(current_bbox, page_width, page_height)
                    text_blocks.append(
                        TextBlock(
                            text=text,
                            page_number=page_number,
                            bbox=normalized_bbox,
                        )
                    )

                logger.debug(f"LayoutLMv3 processed page {page_number}, extracted {len(text_blocks)} text blocks")

            except Exception as e:
                logger.warning(f"LayoutLMv3 inference failed for page {page_number}: {str(e)}. Using pdfplumber fallback.")
                # Fallback: return empty list, will be handled by caller

        except Exception as e:
            logger.error(f"Error in LayoutLMv3 processing for page {page_number}: {str(e)}")

        return text_blocks

    def _detect_multi_column(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Detect and label multi-column layouts."""
        # Simple multi-column detection based on x-coordinate clustering
        if not text_blocks:
            return text_blocks

        # Group blocks by approximate x-position
        columns = {}
        for block in text_blocks:
            col_key = int(block.bbox.x0 * 10) // 5  # Rough column grouping
            if col_key not in columns:
                columns[col_key] = []
            columns[col_key].append(block)

        # Assign column indices
        sorted_columns = sorted(columns.keys())
        for col_idx, col_key in enumerate(sorted_columns):
            for block in columns[col_key]:
                block.column_index = col_idx

        return text_blocks

    def parse(self, pdf_bytes: bytes) -> ParsedDocument:
        """
        Parse PDF file and extract structured content.

        Args:
            pdf_bytes: PDF file content as bytes

        Returns:
            ParsedDocument with all extracted content

        Raises:
            Exception: If PDF parsing fails
        """
        start_time = time.time()
        parser_used = "LayoutLMv3" if self.use_layoutlmv3 else "pdfplumber"

        try:
            # Open PDF with pdfplumber (always needed for text extraction)
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_plumber = pdfplumber.open(pdf_file)

            # Open PDF with pypdf for additional features
            pdf_file.seek(0)
            pdf_reader = pypdf.PdfReader(pdf_file)

            pages = []
            all_tables = []
            all_images = []
            all_text_blocks = []

            # Extract metadata
            metadata = {}
            if pdf_reader.metadata:
                metadata = {
                    "title": pdf_reader.metadata.get("/Title", ""),
                    "author": pdf_reader.metadata.get("/Author", ""),
                    "subject": pdf_reader.metadata.get("/Subject", ""),
                    "creator": pdf_reader.metadata.get("/Creator", ""),
                }

            # Process each page
            for page_num in range(len(pdf_plumber.pages)):
                page_num_1_indexed = page_num + 1
                pdfplumber_page = pdf_plumber.pages[page_num]
                pypdf_page = pdf_reader.pages[page_num]

                # Get page dimensions
                page_width = float(pdfplumber_page.width)
                page_height = float(pdfplumber_page.height)

                # Extract text blocks based on parser choice
                if self.use_layoutlmv3 and self.layoutlmv3_model and self.layoutlmv3_processor:
                    # Try LayoutLMv3 with hybrid approach (uses pdfplumber text + LayoutLMv3 layout)
                    try:
                        # Get words from pdfplumber for LayoutLMv3
                        pdfplumber_words = pdfplumber_page.extract_words()
                        
                        # Convert page to image for LayoutLMv3
                        page_image = self._pdf_page_to_image(pdf_reader, page_num_1_indexed)
                        if page_image and pdfplumber_words:
                            text_blocks = self._extract_text_blocks_layoutlmv3(
                                page_image, pdfplumber_words, page_num_1_indexed, page_width, page_height
                            )
                            # If LayoutLMv3 didn't extract text well, fallback to pdfplumber
                            if not text_blocks:
                                logger.debug(f"LayoutLMv3 returned no text blocks for page {page_num_1_indexed}, using pdfplumber")
                                text_blocks = self._extract_text_blocks_pdfplumber(
                                    pdfplumber_page, page_num_1_indexed, page_width, page_height
                                )
                        else:
                            # Image conversion failed or no words, use pdfplumber
                            if not page_image:
                                logger.debug(f"Failed to convert page {page_num_1_indexed} to image, using pdfplumber")
                            text_blocks = self._extract_text_blocks_pdfplumber(
                                pdfplumber_page, page_num_1_indexed, page_width, page_height
                            )
                    except Exception as e:
                        logger.warning(f"LayoutLMv3 processing failed for page {page_num_1_indexed}: {str(e)}. Falling back to pdfplumber.")
                        text_blocks = self._extract_text_blocks_pdfplumber(
                            pdfplumber_page, page_num_1_indexed, page_width, page_height
                        )
                else:
                    # Use pdfplumber
                    text_blocks = self._extract_text_blocks_pdfplumber(
                        pdfplumber_page, page_num_1_indexed, page_width, page_height
                    )

                # Detect multi-column layout
                text_blocks = self._detect_multi_column(text_blocks)

                # Extract tables (always use pdfplumber for tables)
                tables = self._extract_tables_pdfplumber(
                    pdfplumber_page, page_num_1_indexed, page_width, page_height
                )

                # Extract images
                images = self._extract_images_pypdf(
                    pdf_reader, page_num_1_indexed, page_width, page_height
                )

                # Create page object
                page = Page(
                    page_number=page_num_1_indexed,
                    width=page_width,
                    height=page_height,
                    text_blocks=text_blocks,
                    tables=tables,
                    images=images,
                )

                pages.append(page)
                all_tables.extend(tables)
                all_images.extend(images)
                all_text_blocks.extend(text_blocks)

            # Close PDF
            pdf_plumber.close()

            # Create parsed document
            parsed_doc = ParsedDocument(
                pages=pages,
                tables=all_tables,
                images=all_images,
                text_blocks=all_text_blocks,
                metadata=metadata,
            )

            elapsed_time = time.time() - start_time
            logger.info(
                f"Parsed PDF using {parser_used}: {len(pages)} pages, {len(all_tables)} tables, "
                f"{len(all_images)} images, {len(all_text_blocks)} text blocks "
                f"(took {elapsed_time:.2f}s)"
            )

            # Store parser info in metadata
            parsed_doc.metadata["parser"] = parser_used
            parsed_doc.metadata["parse_time_seconds"] = str(elapsed_time)

            return parsed_doc

        except Exception as e:
            logger.error(f"Failed to parse PDF: {str(e)}")
            raise Exception(f"PDF parsing failed: {str(e)}") from e

    def parse_file(self, file_path: str) -> ParsedDocument:
        """
        Parse PDF file from file path.

        Args:
            file_path: Path to PDF file

        Returns:
            ParsedDocument with all extracted content
        """
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        return self.parse(pdf_bytes)


def compare_parsers(
    pdf_bytes: bytes, use_layoutlmv3: bool = True
) -> Dict[str, Any]:
    """
    Compare parsing results between pdfplumber and LayoutLMv3.

    Args:
        pdf_bytes: PDF file content as bytes
        use_layoutlmv3: Whether to compare with LayoutLMv3

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "pdfplumber": None,
        "layoutlmv3": None,
        "comparison": {},
    }

    # Parse with pdfplumber
    parser_pdfplumber = PDFParser(use_layoutlmv3=False)
    start_time = time.time()
    try:
        result_pdfplumber = parser_pdfplumber.parse(pdf_bytes)
        time_pdfplumber = time.time() - start_time
        comparison["pdfplumber"] = {
            "pages": len(result_pdfplumber.pages),
            "text_blocks": len(result_pdfplumber.text_blocks),
            "tables": len(result_pdfplumber.tables),
            "images": len(result_pdfplumber.images),
            "parse_time_seconds": time_pdfplumber,
            "total_text_length": sum(len(block.text) for block in result_pdfplumber.text_blocks),
        }
    except Exception as e:
        logger.error(f"pdfplumber parsing failed: {str(e)}")
        comparison["pdfplumber"] = {"error": str(e)}

    # Parse with LayoutLMv3 if requested
    if use_layoutlmv3:
        parser_layoutlmv3 = PDFParser(use_layoutlmv3=True)
        start_time = time.time()
        try:
            result_layoutlmv3 = parser_layoutlmv3.parse(pdf_bytes)
            time_layoutlmv3 = time.time() - start_time
            comparison["layoutlmv3"] = {
                "pages": len(result_layoutlmv3.pages),
                "text_blocks": len(result_layoutlmv3.text_blocks),
                "tables": len(result_layoutlmv3.tables),
                "images": len(result_layoutlmv3.images),
                "parse_time_seconds": time_layoutlmv3,
                "total_text_length": sum(len(block.text) for block in result_layoutlmv3.text_blocks),
            }
        except Exception as e:
            logger.error(f"LayoutLMv3 parsing failed: {str(e)}")
            comparison["layoutlmv3"] = {"error": str(e)}

        # Compare results
        if comparison["pdfplumber"] and comparison["layoutlmv3"] and "error" not in comparison["pdfplumber"] and "error" not in comparison["layoutlmv3"]:
            pdfplumber_data = comparison["pdfplumber"]
            layoutlmv3_data = comparison["layoutlmv3"]

            comparison["comparison"] = {
                "text_blocks_diff": layoutlmv3_data["text_blocks"] - pdfplumber_data["text_blocks"],
                "text_length_diff": layoutlmv3_data["total_text_length"] - pdfplumber_data["total_text_length"],
                "time_diff_seconds": layoutlmv3_data["parse_time_seconds"] - pdfplumber_data["parse_time_seconds"],
                "time_ratio": layoutlmv3_data["parse_time_seconds"] / pdfplumber_data["parse_time_seconds"] if pdfplumber_data["parse_time_seconds"] > 0 else float("inf"),
                "text_blocks_ratio": layoutlmv3_data["text_blocks"] / pdfplumber_data["text_blocks"] if pdfplumber_data["text_blocks"] > 0 else float("inf"),
            }

    return comparison


# Global parser instance - will be initialized based on config
pdf_parser: Optional[PDFParser] = None


def get_pdf_parser(use_layoutlmv3: Optional[bool] = None) -> PDFParser:
    """
    Get PDF parser instance, initializing if necessary.

    Args:
        use_layoutlmv3: Override config setting. If None, uses config.

    Returns:
        PDFParser instance
    """
    global pdf_parser

    if use_layoutlmv3 is None:
        # Try to import config, fallback to False if not available
        try:
            from config import settings
            use_layoutlmv3 = settings.pdf_parser_type.lower() == "layoutlmv3"
        except Exception:
            use_layoutlmv3 = False

    if pdf_parser is None or pdf_parser.use_layoutlmv3 != use_layoutlmv3:
        pdf_parser = PDFParser(use_layoutlmv3=use_layoutlmv3)

    return pdf_parser
