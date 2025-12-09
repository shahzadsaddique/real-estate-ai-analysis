"""
Analysis API routes.

This module provides endpoints for retrieving and generating property analysis.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models.analysis import AnalysisStatus
from services import analysis_service, firestore_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class AnalysisResponse(BaseModel):
    """Response model for analysis retrieval."""

    analysis_id: str
    document_id: str
    status: str
    result: Optional[dict] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "analysis_123456",
                "document_id": "doc_123456",
                "status": "complete",
                "result": {
                    "analysis": {
                        "property_address": "123 Main St",
                        "zoning_classification": "R-1",
                    },
                    "source_documents": ["doc_123456"],
                    "confidence_score": 0.95,
                },
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:10:00Z",
                "completed_at": "2024-01-01T00:10:00Z",
            }
        }


class GenerateAnalysisRequest(BaseModel):
    """Request model for analysis generation."""

    document_id: str = Field(..., description="Document ID to analyze")
    document_type: Optional[str] = Field(
        None, description="Document type (zoning, risk, permit, other)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "document_type": "zoning",
            }
        }


class GenerateAnalysisResponse(BaseModel):
    """Response model for analysis generation trigger."""

    analysis_id: str = Field(..., description="Analysis ID")
    document_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Current analysis status")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "analysis_123456",
                "document_id": "doc_123456",
                "status": "processing",
                "message": "Analysis generation started",
            }
        }


# ============================================================================
# API Endpoints
# ============================================================================


@router.get(
    "/documents/{document_id}/analysis",
    summary="Get Document Analysis",
    description="Retrieve generated analysis for a document",
    responses={
        200: {"description": "Analysis retrieved successfully", "model": AnalysisResponse},
        202: {"description": "Analysis is still processing", "model": AnalysisResponse},
        404: {"description": "Analysis not found"},
    },
    tags=["Analysis"],
)
async def get_document_analysis(document_id: str):
    """
    Get analysis for a document.

    This endpoint:
    - Returns analysis if available and complete
    - Returns 202 Accepted if analysis is still processing
    - Returns 404 Not Found if analysis doesn't exist

    Args:
        document_id: Document ID

    Returns:
        AnalysisResponse with analysis data or status

    Raises:
        HTTPException: If analysis not found or error occurs
    """
    try:
        # Services are initialized at startup, but ensure they're ready
        if not firestore_service._client:
            await firestore_service.initialize()

        # Get document to verify it exists
        document = await firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Get latest analysis for document
        analysis = await analysis_service.get_analysis(document_id)

        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis not found for document: {document_id}",
            )

        # Check if analysis is still processing
        if analysis.status in [AnalysisStatus.PROCESSING, AnalysisStatus.PENDING]:
            # Return 202 Accepted for processing status
            response_data = AnalysisResponse(
                analysis_id=analysis.id,
                document_id=analysis.document_id,
                status=analysis.status.value,
                created_at=analysis.created_at.isoformat(),
                updated_at=analysis.updated_at.isoformat(),
            )
            return JSONResponse(
                content=response_data.model_dump(),
                status_code=status.HTTP_202_ACCEPTED,
            )

        # Analysis is complete or failed
        return AnalysisResponse(
            analysis_id=analysis.id,
            document_id=analysis.document_id,
            status=analysis.status.value,
            result=analysis.result.model_dump() if analysis.result else None,
            created_at=analysis.created_at.isoformat(),
            updated_at=analysis.updated_at.isoformat(),
            completed_at=analysis.completed_at.isoformat() if analysis.completed_at else None,
            error_message=analysis.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis",
        )


@router.post(
    "/generate",
    response_model=GenerateAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate Analysis",
    description="""
    Trigger AI-powered analysis generation for a document.
    
    This endpoint initiates the analysis pipeline which:
    - Retrieves relevant document chunks from Pinecone
    - Uses LangChain orchestration with Vertex AI (Gemini Pro)
    - Generates comprehensive property analysis
    - Stores results in Firestore
    
    **Prerequisites**:
    - Document must be fully processed (status: 'indexed' or 'complete')
    - Document chunks must be indexed in Pinecone
    
    **Processing Time**: Analysis typically takes 30-120 seconds depending on document complexity.
    
    **Example Request**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/generate" \\
      -H "Content-Type: application/json" \\
      -d '{
        "document_id": "doc_abc123",
        "document_type": "zoning"
      }'
    ```
    
    **Example Response**:
    ```json
    {
      "analysis_id": "analysis_xyz789",
      "document_id": "doc_abc123",
      "status": "processing",
      "message": "Analysis generation started. Use GET endpoint to check status."
    }
    ```
    """,
    responses={
        202: {
            "description": "Analysis generation started or already exists",
            "model": GenerateAnalysisResponse,
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "analysis_xyz789",
                        "document_id": "doc_abc123",
                        "status": "processing",
                        "message": "Analysis generation started. Use GET endpoint to check status."
                    }
                }
            }
        },
        400: {
            "description": "Document not ready for analysis",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Document is not ready for analysis. Current status: processing. Document must be indexed first."
                    }
                }
            }
        },
        404: {
            "description": "Document not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Document not found: doc_abc123"
                    }
                }
            }
        },
        409: {
            "description": "Analysis already in progress",
            "content": {
                "application/json": {
                    "example": {
                        "analysis_id": "analysis_xyz789",
                        "document_id": "doc_abc123",
                        "status": "processing",
                        "message": "Analysis is already in progress. Use GET endpoint to check status."
                    }
                }
            }
        },
    },
    tags=["Analysis"],
)
async def generate_analysis(request: GenerateAnalysisRequest):
    """
    Generate AI-powered analysis for a document.

    This endpoint triggers the complete analysis pipeline:
    1. Validates document is ready (indexed/complete)
    2. Checks for existing analysis (returns if complete/in-progress)
    3. Starts async analysis generation using LangChain + Vertex AI
    4. Returns immediately with analysis ID and status

    **Analysis Process**:
    - Retrieves relevant document chunks via semantic search (Pinecone)
    - Uses multi-step reasoning with LangChain
    - Generates structured JSON analysis using Vertex AI Gemini Pro
    - Extracts property information, zoning, risks, permits, recommendations
    - Stores complete analysis in Firestore

    Args:
        request: GenerateAnalysisRequest containing:
            - document_id (required): Document ID to analyze
            - document_type (optional): Document type hint for better analysis
                Valid values: 'zoning', 'risk', 'permit', 'other'

    Returns:
        GenerateAnalysisResponse containing:
            - analysis_id: Unique identifier for the analysis
            - document_id: Document ID being analyzed
            - status: Current status ('processing', 'complete', 'pending')
            - message: Human-readable status message

    Raises:
        HTTPException 400: If document is not ready for analysis (not indexed)
        HTTPException 404: If document not found
        HTTPException 500: If analysis generation fails

    Example:
        ```python
        import requests
        
        response = requests.post(
            'http://localhost:8000/api/v1/generate',
            json={
                'document_id': 'doc_abc123',
                'document_type': 'zoning'
            }
        )
        result = response.json()
        print(f"Analysis ID: {result['analysis_id']}")
        print(f"Status: {result['status']}")
        ```

    Note:
        - Analysis generation is asynchronous
        - Use GET /documents/{document_id}/analysis to check status
        - Analysis typically completes in 30-120 seconds
        - Results include property details, zoning info, risks, permits, recommendations
    """
    try:
        # Services are initialized at startup, but ensure they're ready
        if not firestore_service._client:
            await firestore_service.initialize()
        # Analysis service initialization is handled internally

        document_id = request.document_id
        document_type = request.document_type or "other"

        # Verify document exists
        document = await firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Check if analysis already exists
        existing_analysis = await analysis_service.get_analysis(document_id)

        if existing_analysis:
            if existing_analysis.status == AnalysisStatus.COMPLETE:
                # Analysis already complete
                return GenerateAnalysisResponse(
                    analysis_id=existing_analysis.id,
                    document_id=document_id,
                    status=existing_analysis.status.value,
                    message="Analysis already completed. Use GET endpoint to retrieve results.",
                )
            elif existing_analysis.status == AnalysisStatus.PROCESSING:
                # Analysis already in progress
                return GenerateAnalysisResponse(
                    analysis_id=existing_analysis.id,
                    document_id=document_id,
                    status=existing_analysis.status.value,
                    message="Analysis is already in progress. Use GET endpoint to check status.",
                )
            elif existing_analysis.status == AnalysisStatus.FAILED:
                # Previous analysis failed, allow retry
                logger.info(
                    f"Previous analysis failed for document {document_id}. "
                    "Starting new analysis generation."
                )
            # If PENDING, continue to generate

        # Check if document is ready for analysis
        if document.status.value not in ["indexed", "complete"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document is not ready for analysis. Current status: {document.status.value}. "
                "Document must be indexed first.",
            )

        # Start analysis generation (async - don't wait for completion)
        # In production, this would be triggered via Pub/Sub or background task
        # For now, we'll start it and return immediately
        try:
            # Generate analysis asynchronously
            # Note: In production, this should be triggered via Pub/Sub or background worker
            import asyncio

            # Start analysis generation in background
            asyncio.create_task(
                analysis_service.generate_analysis(document_id, document_type)
            )

            # Start analysis generation in background task
            # This allows the endpoint to return immediately while analysis runs
            task = asyncio.create_task(
                analysis_service.generate_analysis(document_id, document_type)
            )

            # Wait a moment to get the analysis ID from the created record
            # In production, this would be handled via Pub/Sub or a job queue
            await asyncio.sleep(0.1)  # Brief wait for analysis record creation

            # Get the newly created analysis
            new_analysis = await analysis_service.get_analysis(document_id)

            if new_analysis:
                analysis_id = new_analysis.id
                current_status = new_analysis.status.value
            else:
                # Fallback if analysis not found yet
                analysis_id = f"analysis_{document_id}_{int(datetime.now(timezone.utc).timestamp())}"
                current_status = "processing"

            return GenerateAnalysisResponse(
                analysis_id=analysis_id,
                document_id=document_id,
                status=current_status,
                message="Analysis generation started. Use GET endpoint to check status.",
            )

        except Exception as e:
            logger.error(f"Failed to start analysis generation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start analysis generation: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate analysis",
        )
