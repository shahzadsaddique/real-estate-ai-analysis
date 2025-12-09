# Real Estate AI Analysis Platform

A production-ready MVP demonstrating expertise in Python FastAPI, Google Cloud Platform, prompt engineering, and LLM orchestration for real estate document analysis.

## 🎯 Project Overview

This platform ingests complex real estate documents (zoning maps, risk assessments, permits) and generates professional-grade property analysis using frontier LLM models. It showcases:

- **FastAPI**: Scalable async API with proper error handling
- **GCP Integration**: Cloud Run, Pub/Sub, Firestore, Vertex AI
- **Advanced Prompt Engineering**: System prompts with complex logic and JSON formatting
- **Intelligent Document Processing**: Layout-aware PDF chunking preserving context
- **Production Architecture**: Async workflows, monitoring, and scalability

## 📚 Documentation

- **[PROJECT_OVERVIEW.md](./PROJECT_OVERVIEW.md)**: Complete project overview and goals
- **[ARCHITECTURE.md](./ARCHITECTURE.md)**: Technical architecture with Mermaid diagrams

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (for backend)
- **Node.js 20+** (for frontend)
- **Google Cloud Platform account** with billing enabled
- **Pinecone account** (free tier available)
- **Docker** (for containerized deployment)
- **Google Cloud SDK** (gcloud CLI) for deployment

### Local Development Setup

#### 1. Clone Repository
```bash
git clone <repository-url>
cd realestate-demo
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials (see Environment Variables section below)

# Run the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.example .env.local
# Edit .env.local with your API URL (see Environment Variables section below)

# Run development server
npm run dev
```

The frontend will be available at:
- **Frontend**: http://localhost:3000

#### 4. Worker Setup (Required for Document Processing)

The worker processes documents asynchronously from Pub/Sub messages. **You must run the worker for documents to be processed after upload.**

**Prerequisites:**
1. Pub/Sub topic must exist (created during backend setup)
2. Pub/Sub subscription must exist for the worker

**Create Pub/Sub Subscription (if not exists):**
```bash
# Set your project ID
export PROJECT_ID=your-gcp-project-id

# Create subscription for the worker
# The subscription name should match: {PUBSUB_TOPIC_NAME}-sub
gcloud pubsub subscriptions create document-processing-sub \
  --topic=document-processing \
  --project=$PROJECT_ID \
  --ack-deadline=600 \
  --message-retention-duration=7d
```

**Run Worker Locally:**
```bash
cd backend

# Ensure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Ensure .env is configured (same as backend API)
# The worker uses the same environment variables

# Run the worker
python -m workers.main
```

**Worker Output:**
You should see:
```
INFO - Worker initialized for subscription: projects/your-project/subscriptions/document-processing-sub
INFO - Starting document processing worker...
INFO - Listening for messages on projects/your-project/subscriptions/document-processing-sub...
Press Ctrl+C to stop.
```

**Verify Worker is Processing:**
1. Upload a document via the API (see Document Management endpoints)
2. Check worker logs - you should see:
   ```
   INFO - Received message for document: doc_xxxxx
   INFO - Downloading document doc_xxxxx from Cloud Storage
   INFO - Parsing PDF for document doc_xxxxx
   ...
   ```
3. Check document status:
   ```bash
   curl http://localhost:8000/api/v1/documents/{document_id}/status
   ```

**Note:** The worker must be running continuously to process documents. For production, deploy it as a Cloud Run service (see Deployment section).

### Environment Variables

#### Backend Environment Variables (`.env`)

Create a `.env` file in the `backend/` directory with the following variables:

```bash
# Google Cloud Platform Configuration
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
FIRESTORE_DATABASE_ID=(default)

# Pinecone Vector Database
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=realestate-documents

# Google Cloud Pub/Sub
PUBSUB_TOPIC_NAME=document-processing

# Google Cloud Storage
STORAGE_BUCKET_NAME=realestate-documents

# Vertex AI Configuration
VERTEX_AI_MODEL_NAME=gemini-2.5-pro
VERTEX_AI_LOCATION=us-central1

# API Security
API_SECRET_KEY=your-secret-key-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Logging
LOG_LEVEL=INFO
```

**Getting Your Credentials:**

1. **GCP Project ID**: From Google Cloud Console
2. **Pinecone API Key**: From [Pinecone Console](https://app.pinecone.io/)
3. **GCP Credentials**: 
   ```bash
   gcloud auth application-default login
   ```
4. **Storage Bucket**: Create in GCP Console or via:
   ```bash
   gsutil mb -p $GCP_PROJECT_ID -l us-central1 gs://realestate-documents
   ```
5. **Pub/Sub Topic and Subscription**: Create via:
   ```bash
   # Create topic
   gcloud pubsub topics create document-processing --project=$GCP_PROJECT_ID
   
   # Create subscription for worker (required for document processing)
   gcloud pubsub subscriptions create document-processing-sub \
     --topic=document-processing \
     --project=$GCP_PROJECT_ID \
     --ack-deadline=600 \
     --message-retention-duration=7d
   ```

#### Frontend Environment Variables (`.env.local`)

Create a `.env.local` file in the `frontend/` directory:

```bash
# API URL (local development)
NEXT_PUBLIC_API_URL=http://localhost:8000

# For production, use your deployed API URL:
# NEXT_PUBLIC_API_URL=https://realestate-api-xxxxx-uc.a.run.app
```

## 🏗️ Architecture Highlights

### Key Components

1. **FastAPI Application**: RESTful API with async document upload
2. **Pub/Sub Worker**: Async document processing pipeline
3. **LayoutLMv3 Parser**: Intelligent PDF parsing preserving layout
4. **Intelligent Chunker**: Context-aware chunking for complex layouts
5. **LangChain Orchestration**: Multi-step reasoning with Vertex AI
6. **Pinecone Vector DB**: Semantic search for document retrieval
7. **Next.js Frontend**: Modern UI for document management

### Technology Stack

**Backend:**
- FastAPI (async API framework)
- LangChain (LLM orchestration)
- Vertex AI (Gemini Pro for LLM)
- Pinecone (vector database)
- Firestore (metadata storage)
- Pub/Sub (async messaging)
- Cloud Storage (document storage)

**Frontend:**
- Next.js 14 (React framework)
- TypeScript
- Tailwind CSS
- Shadcn/ui

**Infrastructure:**
- Google Cloud Run
- Cloud Build (CI/CD)
- Cloud Logging & Monitoring

## 📋 Features

### Document Processing
- ✅ PDF upload with validation
- ✅ Layout-aware parsing (tables, images, multi-column)
- ✅ Intelligent chunking preserving context
- ✅ Vector embedding generation
- ✅ Semantic search capabilities

### Analysis Generation
- ✅ Multi-document synthesis
- ✅ Professional JSON-formatted output
- ✅ LangChain orchestration
- ✅ Vertex AI integration
- ✅ Context preservation across chunks

**Built for demonstrating expertise in Python FastAPI, GCP, and LLM orchestration for real estate AI applications.**

