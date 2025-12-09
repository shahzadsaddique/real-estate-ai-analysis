"""
Cloud Storage service for file operations.

This module provides methods for uploading, downloading, and managing files
in Google Cloud Storage.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.cloud.storage import Bucket, Client

from config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Cloud Storage service for file operations."""

    def __init__(self):
        """Initialize Storage service."""
        self._client: Optional[Client] = None
        self._bucket: Optional[Bucket] = None
        self._bucket_name: str = settings.storage_bucket_name

    def initialize(self):
        """Initialize Cloud Storage client and bucket connection."""
        if self._client is not None:
            # Already initialized, skip
            return
        
        try:
            self._client = storage.Client(project=settings.gcp_project_id)
            self._bucket = self._client.bucket(self._bucket_name)

            # Verify bucket exists
            if not self._bucket.exists():
                logger.warning(
                    f"Bucket {self._bucket_name} does not exist. "
                    "It will be created on first upload."
                )

            logger.info(
                f"Cloud Storage client initialized for bucket: {self._bucket_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Storage client: {str(e)}")
            raise

    def close(self):
        """Close Cloud Storage client connection."""
        # Cloud Storage client doesn't need explicit closing
        logger.info("Cloud Storage client closed")

    @property
    def client(self) -> Client:
        """Get Cloud Storage client."""
        if not self._client:
            raise RuntimeError(
                "Cloud Storage client not initialized. Call initialize() first."
            )
        return self._client

    @property
    def bucket(self) -> Bucket:
        """Get Cloud Storage bucket."""
        if not self._bucket:
            raise RuntimeError(
                "Cloud Storage bucket not initialized. Call initialize() first."
            )
        return self._bucket

    def upload_file(
        self,
        file: bytes,
        path: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a file to Cloud Storage.

        Args:
            file: File content as bytes
            path: Storage path (e.g., "documents/doc_123.pdf")
            content_type: MIME type (e.g., "application/pdf")
            metadata: Optional metadata dictionary

        Returns:
            GCS path (gs://bucket/path)

        Raises:
            RuntimeError: If client is not initialized
            Exception: If upload fails
        """
        if not self._client:
            self.initialize()

        try:
            # Get or create blob
            blob = self.bucket.blob(path)

            # Set content type
            blob.content_type = content_type

            # Set metadata if provided
            if metadata:
                blob.metadata = metadata

            # Add upload timestamp to metadata
            if not blob.metadata:
                blob.metadata = {}
            blob.metadata["uploaded_at"] = datetime.now(timezone.utc).isoformat()

            # Upload file
            blob.upload_from_string(file, content_type=content_type)

            gcs_path = f"gs://{self._bucket_name}/{path}"
            logger.info(f"Uploaded file to: {gcs_path}")

            return gcs_path

        except Exception as e:
            logger.error(f"Failed to upload file to {path}: {str(e)}")
            raise

    def download_file(self, path: str) -> bytes:
        """
        Download a file from Cloud Storage.

        Args:
            path: Storage path (e.g., "documents/doc_123.pdf")

        Returns:
            File content as bytes

        Raises:
            RuntimeError: If client is not initialized
            NotFound: If file does not exist
            Exception: If download fails
        """
        if not self._client:
            self.initialize()

        try:
            blob = self.bucket.blob(path)

            # Check if blob exists
            if not blob.exists():
                logger.warning(f"File not found: {path}")
                raise NotFound(f"File not found: {path}")

            # Download file
            file_content = blob.download_as_bytes()

            logger.info(f"Downloaded file from: {path}")
            return file_content

        except NotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to download file from {path}: {str(e)}")
            raise

    def generate_signed_url(
        self, path: str, expiration: int = 3600, method: str = "GET"
    ) -> str:
        """
        Generate a signed URL for secure file access.

        Args:
            path: Storage path (e.g., "documents/doc_123.pdf")
            expiration: URL expiration time in seconds (default: 1 hour)
            method: HTTP method allowed (default: "GET")

        Returns:
            Signed URL string

        Raises:
            RuntimeError: If client is not initialized
            Exception: If URL generation fails
        """
        if not self._client:
            self.initialize()

        try:
            blob = self.bucket.blob(path)

            # Check if blob exists
            if not blob.exists():
                logger.warning(f"File not found for signed URL: {path}")
                raise NotFound(f"File not found: {path}")

            # Generate signed URL
            url = blob.generate_signed_url(
                expiration=timedelta(seconds=expiration),
                method=method,
                version="v4",
            )

            logger.info(f"Generated signed URL for: {path} (expires in {expiration}s)")
            return url

        except NotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {path}: {str(e)}")
            raise

    def delete_file(self, path: str) -> bool:
        """
        Delete a file from Cloud Storage.

        Args:
            path: Storage path (e.g., "documents/doc_123.pdf")

        Returns:
            True if successful, False if file doesn't exist

        Raises:
            RuntimeError: If client is not initialized
            Exception: If deletion fails
        """
        if not self._client:
            self.initialize()

        try:
            blob = self.bucket.blob(path)

            # Check if blob exists
            if not blob.exists():
                logger.warning(f"File not found for deletion: {path}")
                return False

            # Delete file
            blob.delete()

            logger.info(f"Deleted file: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file {path}: {str(e)}")
            raise

    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in Cloud Storage.

        Args:
            path: Storage path (e.g., "documents/doc_123.pdf")

        Returns:
            True if file exists, False otherwise

        Raises:
            RuntimeError: If client is not initialized
        """
        if not self._client:
            self.initialize()

        try:
            blob = self.bucket.blob(path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check file existence for {path}: {str(e)}")
            return False

    def get_file_metadata(self, path: str) -> Dict[str, Any]:
        """
        Get file metadata from Cloud Storage.

        Args:
            path: Storage path (e.g., "documents/doc_123.pdf")

        Returns:
            Dictionary with file metadata

        Raises:
            RuntimeError: If client is not initialized
            NotFound: If file does not exist
            Exception: If metadata retrieval fails
        """
        if not self._client:
            self.initialize()

        try:
            blob = self.bucket.blob(path)

            # Check if blob exists
            if not blob.exists():
                logger.warning(f"File not found: {path}")
                raise NotFound(f"File not found: {path}")

            # Reload blob to get metadata
            blob.reload()

            metadata = {
                "path": path,
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "time_created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "md5_hash": blob.md5_hash,
                "etag": blob.etag,
                "metadata": blob.metadata or {},
            }

            logger.debug(f"Retrieved metadata for: {path}")
            return metadata

        except NotFound:
            raise
        except Exception as e:
            logger.error(f"Failed to get file metadata for {path}: {str(e)}")
            raise

    def list_files(self, prefix: str = "", max_results: Optional[int] = None) -> List[str]:
        """
        List files in Cloud Storage with optional prefix filter.

        Args:
            prefix: Optional prefix to filter files (e.g., "documents/")
            max_results: Optional maximum number of results

        Returns:
            List of file paths

        Raises:
            RuntimeError: If client is not initialized
            Exception: If listing fails
        """
        if not self._client:
            self.initialize()

        try:
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
            file_paths = [blob.name for blob in blobs]

            logger.info(f"Listed {len(file_paths)} files with prefix: {prefix}")
            return file_paths

        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {str(e)}")
            raise

    def copy_file(self, source_path: str, destination_path: str) -> str:
        """
        Copy a file within Cloud Storage.

        Args:
            source_path: Source file path
            destination_path: Destination file path

        Returns:
            GCS path of copied file (gs://bucket/path)

        Raises:
            RuntimeError: If client is not initialized
            NotFound: If source file does not exist
            Exception: If copy fails
        """
        if not self._client:
            self.initialize()

        try:
            source_blob = self.bucket.blob(source_path)

            # Check if source exists
            if not source_blob.exists():
                logger.warning(f"Source file not found: {source_path}")
                raise NotFound(f"Source file not found: {source_path}")

            # Copy blob
            new_blob = self.bucket.copy_blob(source_blob, self.bucket, destination_path)

            gcs_path = f"gs://{self._bucket_name}/{destination_path}"
            logger.info(f"Copied file from {source_path} to {destination_path}")

            return gcs_path

        except NotFound:
            raise
        except Exception as e:
            logger.error(
                f"Failed to copy file from {source_path} to {destination_path}: {str(e)}"
            )
            raise


# Global service instance
storage_service = StorageService()
