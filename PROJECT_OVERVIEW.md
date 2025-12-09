# Real Estate AI Analysis Platform - Project Overview

## Executive Summary

This MVP demonstrates a production-ready real estate intelligence system that ingests complex property documents (zoning maps, risk assessments, permits) and generates professional-grade property analysis using frontier LLM models. The system showcases expertise in Python FastAPI, GCP services, prompt engineering, and LLM orchestration.

## Project Goals

1. **Demonstrate FastAPI Expertise**: Build a scalable, async API with proper error handling, validation, and documentation
2. **Showcase GCP Integration**: Seamless integration with Cloud Run, Pub/Sub, Firestore, and Vertex AI
3. **Advanced Prompt Engineering**: System prompts that handle complex logic, JSON formatting, and multi-step reasoning
4. **Intelligent Document Processing**: Handle PDFs with complex layouts (tables, images) without losing context
5. **Production-Ready Architecture**: Async workflows, proper error handling, monitoring, and scalability

## Core Features

### 1. Document Ingestion Pipeline
- **PDF Processing**: Advanced chunking strategy for complex layouts (tables, images, multi-column)
- **LayoutLMv3 Parser**: Custom parser for extracting structured data from real estate documents
- **Async Pub/Sub Flow**: Non-blocking document ingestion with proper error handling

### 2. Vector Search & Retrieval
- **Pinecone Integration**: Store and retrieve document embeddings
- **Hybrid Search**: Combine semantic search with metadata filtering
- **Context Preservation**: Maintain document structure and relationships in chunks

### 3. LLM Orchestration
- **Vertex AI Integration**: Use Google's Vertex AI for LLM inference
- **LangChain Framework**: Orchestrate complex multi-step reasoning workflows
- **System Prompts**: Professional-grade prompts for property analysis generation

### 4. Property Analysis Generation
- **Structured Output**: JSON-formatted analysis reports
- **Multi-Document Synthesis**: Combine information from zoning, risk, and permit documents
- **Professional Formatting**: Generate publication-ready property analysis

### 5. Frontend Dashboard
- **Next.js Application**: Modern, responsive UI with App Router
- **Document Upload Page** (`/upload`): Drag-and-drop interface with file validation
- **Document List Page** (`/documents`): View all documents with status indicators
- **Home Page** (`/`): Landing page with features and navigation
- **Real-time Status Updates**: Polling-based status checks for document processing
- **Analysis Viewer**: Structured display of property analysis results
- **Document Management**: View, download, and delete documents
- **Signed URL Generation**: Secure access to PDF files in Cloud Storage

## Technology Stack

### Backend
- **Python 3.11+**
- **FastAPI 0.104.1**: Async API framework with automatic OpenAPI docs
- **Uvicorn 0.24.0**: ASGI server
- **LangChain 0.1.0**: LLM orchestration framework
- **LangChain Google Vertex AI 0.0.6**: Vertex AI integration
- **Vertex AI**: LLM inference with Gemini 2.5 Pro model
- **Pinecone 5.0+**: Serverless vector database for embeddings
- **Firestore 2.13.1**: NoSQL database for document metadata
- **Pub/Sub 2.18.4**: Async message queue
- **Cloud Storage 2.14.0**: Document file storage
- **pdfplumber 0.10.3**: PDF parsing (default parser)
- **Transformers 4.35.2**: Optional LayoutLMv3 parser support
- **Pydantic 2.5.0**: Data validation and settings management

### Frontend
- **Next.js 14.0.4**: React framework with App Router
- **React 18.2.0**: UI library
- **TypeScript 5.3.3**: Type safety
- **Tailwind CSS 3.3.6**: Utility-first CSS framework
- **Shadcn/ui**: Accessible UI component library (Radix UI primitives)
- **Axios 1.6.2**: HTTP client
- **Zod 3.22.4**: Schema validation
- **Lucide React 0.294.0**: Icon library

### Infrastructure
- **Google Cloud Run**: Containerized API and workers
- **Cloud Build**: CI/CD pipeline
- **Cloud Logging**: Centralized logging
- **Cloud Monitoring**: Performance metrics

## Project Structure

```
real-estate-ai-analysis/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app entry point
│   │   └── routes/
│   │       ├── documents.py     # Document CRUD endpoints
│   │       └── analysis.py      # Analysis endpoints
│   ├── workers/
│   │   ├── main.py              # Pub/Sub worker entry point
│   │   └── document_processor.py  # Document processing logic
│   ├── services/
│   │   ├── analysis_service.py   # LangChain orchestration
│   │   ├── embedding_service.py # Vertex AI embeddings
│   │   ├── llm_service.py       # LangChain + Vertex AI LLM
│   │   ├── firestore_service.py # Firestore operations
│   │   ├── storage_service.py   # Cloud Storage operations
│   │   ├── pubsub_service.py    # Pub/Sub messaging
│   │   └── pinecone_service.py  # Vector database operations
│   ├── models/
│   │   ├── document.py          # Document data models
│   │   ├── chunk.py             # Chunk data models
│   │   ├── analysis.py          # Analysis data models
│   │   └── user.py              # User data models
│   ├── utils/
│   │   ├── pdf_parser.py        # PDF parsing (pdfplumber/LayoutLMv3)
│   │   └── chunker.py           # Intelligent chunking
│   ├── prompts/
│   │   ├── analysis_prompt.py   # Analysis generation prompts
│   │   ├── formatting_prompt.py # JSON formatting prompts
│   │   └── retrieval_prompt.py # Retrieval prompts
│   ├── config.py                # Pydantic settings
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # API container
│   └── Dockerfile.worker        # Worker container
├── frontend/
│   ├── app/
│   │   ├── layout.tsx           # Root layout
│   │   ├── page.tsx             # Homepage
│   │   ├── globals.css          # Global styles
│   │   ├── documents/
│   │   │   └── page.tsx         # Document list page
│   │   └── upload/
│   │       └── page.tsx         # Upload page
│   ├── components/
│   │   ├── DocumentUpload.tsx   # Upload component
│   │   ├── DocumentList.tsx    # Document list component
│   │   ├── AnalysisViewer.tsx  # Analysis display component
│   │   └── ui/                  # Shadcn/ui components
│   ├── lib/
│   │   ├── api.ts               # API client
│   │   └── utils.ts             # Utilities
│   ├── package.json             # Node dependencies
│   ├── next.config.js           # Next.js config
│   ├── tailwind.config.js       # Tailwind config
│   └── Dockerfile               # Frontend container
└── docs/
    ├── ARCHITECTURE.md          # Technical architecture
    ├── PROJECT_OVERVIEW.md      # This file
    ├── DEPLOYMENT.md            # Deployment guide
    ├── CODE_WALKTHROUGH.md      # Code walkthrough
    └── TEST_DOCUMENTS.md        # Test document guide
```

## API Endpoints

### Document Management
- `POST /api/v1/documents/upload` - Upload PDF document
- `GET /api/v1/documents` - List documents (with optional user_id filter)
- `GET /api/v1/documents/{document_id}` - Get document details
- `GET /api/v1/documents/{document_id}/status` - Get processing status
- `GET /api/v1/documents/{document_id}/download` - Download original PDF
- `DELETE /api/v1/documents/{document_id}` - Delete document

### Analysis
- `GET /api/v1/documents/{document_id}/analysis` - Get analysis results
- `POST /api/v1/generate` - Trigger analysis generation

### System
- `GET /api/v1/health` - Health check
- `GET /docs` - OpenAPI documentation
- `GET /redoc` - ReDoc documentation

## Document Types Supported

The system supports classification and specialized processing for:

1. **Zoning Documents** (`zoning`)
   - Zoning maps and regulations
   - Land use restrictions
   - Building height limits
   - Setback requirements

2. **Risk Assessment Documents** (`risk`)
   - Flood zone maps
   - Fire risk assessments
   - Environmental hazards
   - Geological surveys

3. **Permit Documents** (`permit`)
   - Building permits
   - Zoning variances
   - Environmental permits
   - Construction approvals

4. **Other Documents** (`other`)
   - General property documents
   - Mixed document types
   - Unclassified documents

## Environment Variables

### Backend (.env)

**Required:**
```bash
GCP_PROJECT_ID=your-gcp-project-id
STORAGE_BUCKET_NAME=your-storage-bucket
PINECONE_API_KEY=your-pinecone-api-key
API_SECRET_KEY=your-api-secret-key
```

**Optional:**
```bash
GCP_REGION=us-central1
PINECONE_INDEX_NAME=realestate-documents
PUBSUB_TOPIC_NAME=document-processing
VERTEX_AI_MODEL_NAME=gemini-2.5-pro
CORS_ORIGINS=http://localhost:3000
LOG_LEVEL=INFO
PDF_PARSER_TYPE=pdfplumber
```

**Pinecone Configuration (choose one):**
```bash
# For serverless indexes
HOST=https://your-index.svc.environment.pinecone.io

# For pod-based indexes
PINECONE_ENVIRONMENT=us-east-1
# OR
REGION=us-east-1
```

### Frontend (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
```

## Key Challenges Addressed

### 1. Complex PDF Chunking
- **Problem**: Real estate documents contain tables, images, and multi-column layouts
- **Solution**: Layout-aware chunking with pdfplumber (default) or LayoutLMv3 (optional), preserving spatial relationships
- **Implementation**: Intelligent chunker with table preservation, image+caption handling, and sentence-boundary awareness
- **Demonstration**: Chunks maintain page numbers, element types, and spatial metadata

### 2. LangChain vs LlamaIndex
- **Choice**: LangChain for this project
- **Reasoning**: Better integration with Vertex AI, more flexible orchestration, superior prompt management
- **Implementation**: Multi-step reasoning chains with analysis generation and JSON validation
- **Demonstration**: Two-stage pipeline (analysis → formatting) with LangChain chains

### 3. Async Pub/Sub Handoff
- **Problem**: Non-blocking document processing with proper error handling
- **Solution**: FastAPI async endpoints → Pub/Sub → Cloud Run workers
- **Implementation**: 202 Accepted responses, status polling, error recovery
- **Demonstration**: Async flow with retry logic and status tracking

### 4. GCP Cloud Run Deployment
- **Problem**: Containerized Python services with proper scaling
- **Solution**: Separate Dockerfiles for API and worker, Cloud Run configuration, environment variables
- **Implementation**: Auto-scaling based on CPU and concurrency, separate services for API and workers
- **Demonstration**: Independent scaling of API and worker services

### 5. Vector Search & Retrieval
- **Problem**: Efficient semantic search across large document collections
- **Solution**: Pinecone serverless vector database with metadata filtering
- **Implementation**: Embeddings generated with Vertex AI text-embedding-005, stored in Pinecone with document metadata
- **Fallback**: Firestore-based retrieval if Pinecone unavailable

### 6. LLM Orchestration
- **Problem**: Complex multi-step reasoning for property analysis
- **Solution**: LangChain chains with Vertex AI Gemini 2.5 Pro
- **Implementation**: Analysis chain (reasoning) → Formatting chain (JSON validation)
- **Features**: Temperature control, token limits, retry logic, fallback handling

## Success Metrics

1. **Document Processing**: Handle 100+ page PDFs with complex layouts
2. **Response Time**: Generate analysis in 30-120 seconds for typical documents
3. **Accuracy**: Maintain context across document chunks with semantic search
4. **Scalability**: Handle concurrent document processing with auto-scaling
5. **Reliability**: 99.9% uptime with proper error handling and retry logic
6. **File Size**: Support PDFs up to 50MB
7. **Processing Stages**: Track progress through 8 distinct stages
8. **Analysis Quality**: Structured JSON output with property details, zoning, risks, permits, and recommendations

## Frontend Features

### Document Upload (`/upload`)
- Drag-and-drop file upload
- File type validation (PDF only)
- File size validation (max 50MB)
- Document type selection dropdown
- Upload progress indicator
- Success/error notifications
- Automatic redirect to document list

### Document List (`/documents`)
- List all user documents
- Status badges (uploaded, processing, complete, failed)
- Document type indicators
- File size and metadata display
- Action buttons (view, download, delete)
- Real-time status polling
- Filter by status or type (future)

### Analysis Viewer
- Structured property analysis display
- Property address and type
- Zoning classification and summary
- Risk assessment details
- Permit requirements list
- Restrictions and recommendations
- Key findings highlights
- Confidence score display
- Processing time information

### Home Page (`/`)
- Hero section with value proposition
- Feature highlights
- "How It Works" step-by-step guide
- Technology stack badges
- Call-to-action buttons
- Navigation to upload and documents pages

## Next Steps

1. Review technical architecture document
2. Follow build prompts to implement the system
3. Deploy to GCP using deployment guide
4. Prepare code walkthrough demonstration

