"""
Application configuration using Pydantic Settings.

This module loads and validates environment variables for the application.
All configuration values are type-checked and validated at startup.
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google Cloud Platform Configuration
    gcp_project_id: str = Field(
        ...,
        description="Google Cloud Platform project ID",
        alias="GCP_PROJECT_ID",
    )
    gcp_region: str = Field(
        default="us-central1",
        description="GCP region for services",
        alias="GCP_REGION",
    )
    firestore_database_id: str = Field(
        default="(default)",
        description="Firestore database ID",
        alias="FIRESTORE_DATABASE_ID",
    )

    # Pinecone Vector Database Configuration
    pinecone_api_key: str = Field(
        ...,
        description="Pinecone API key",
        alias="PINECONE_API_KEY",
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment/region (for pod-based indexes)",
        alias="PINECONE_ENVIRONMENT",
    )
    pinecone_host: Optional[str] = Field(
        default=None,
        description="Pinecone serverless index host URL (for serverless indexes)",
        alias="PINECONE_HOST",
    )
    pinecone_region: Optional[str] = Field(
        default=None,
        description="Pinecone region (alternative to environment)",
        alias="PINECONE_REGION",
    )
    pinecone_index_name: str = Field(
        default="realestate-documents",
        description="Pinecone index name for document embeddings",
        alias="PINECONE_INDEX_NAME",
    )

    # Google Cloud Pub/Sub Configuration
    pubsub_topic_name: str = Field(
        default="document-processing",
        description="Pub/Sub topic name for document processing tasks",
        alias="PUBSUB_TOPIC_NAME",
    )

    # Google Cloud Storage Configuration
    storage_bucket_name: str = Field(
        ...,
        description="Cloud Storage bucket name for document storage",
        alias="STORAGE_BUCKET_NAME",
    )

    # Vertex AI Configuration
    vertex_ai_model_name: str = Field(
        default="gemini-2.5-pro",
        description="Vertex AI model name for LLM inference ",
        alias="VERTEX_AI_MODEL_NAME",
    )

    # API Security
    api_secret_key: str = Field(
        ...,
        description="Secret key for API authentication (change in production)",
        alias="API_SECRET_KEY",
    )

    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:3000,https://realestate-frontend-784415003538.us-central1.run.app",
        description="Comma-separated list of allowed CORS origins",
        alias="CORS_ORIGINS",
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        alias="LOG_LEVEL",
    )

    # PDF Parser Configuration
    pdf_parser_type: str = Field(
        default="pdfplumber",
        description="PDF parser type: 'pdfplumber' or 'layoutlmv3'",
        alias="PDF_PARSER_TYPE",
    )

    @field_validator("cors_origins")
    @classmethod
    def parse_cors_origins(cls, v: str) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"LOG_LEVEL must be one of {allowed_levels}")
        return v_upper

    @field_validator("pdf_parser_type")
    @classmethod
    def validate_pdf_parser_type(cls, v: str) -> str:
        """Validate PDF parser type is one of the allowed values."""
        allowed_types = ["pdfplumber", "layoutlmv3"]
        v_lower = v.lower()
        if v_lower not in allowed_types:
            raise ValueError(f"PDF_PARSER_TYPE must be one of {allowed_types}")
        return v_lower

    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return self.parse_cors_origins(self.cors_origins)


# Global settings instance
settings = Settings()
