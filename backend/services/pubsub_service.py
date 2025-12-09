"""
Pub/Sub service for asynchronous message publishing.

This module provides async methods for publishing messages to Google Cloud Pub/Sub
for document processing tasks.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.api_core import retry
from google.cloud.pubsub_v1 import PublisherClient

from config import settings

logger = logging.getLogger(__name__)


class PubSubService:
    """Pub/Sub service for publishing messages."""

    def __init__(self):
        """Initialize Pub/Sub service."""
        self._publisher: Optional[PublisherClient] = None
        self._topic_path: Optional[str] = None

    def initialize(self):
        """Initialize Pub/Sub publisher client and topic path."""
        if self._publisher is not None:
            # Already initialized, skip
            return
        
        try:
            self._publisher = PublisherClient()
            self._topic_path = self._publisher.topic_path(
                settings.gcp_project_id, settings.pubsub_topic_name
            )
            logger.info(
                f"Pub/Sub publisher initialized for topic: {settings.pubsub_topic_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub publisher: {str(e)}")
            raise

    def close(self):
        """Close Pub/Sub publisher client."""
        if self._publisher:
            self._publisher.close()
            logger.info("Pub/Sub publisher closed")

    @property
    def publisher(self) -> PublisherClient:
        """Get Pub/Sub publisher client."""
        if not self._publisher:
            raise RuntimeError(
                "Pub/Sub publisher not initialized. Call initialize() first."
            )
        return self._publisher

    @property
    def topic_path(self) -> str:
        """Get topic path."""
        if not self._topic_path:
            raise RuntimeError(
                "Pub/Sub topic path not initialized. Call initialize() first."
            )
        return self._topic_path

    def _serialize_message(self, data: dict) -> bytes:
        """
        Serialize message data to JSON bytes.

        Args:
            data: Dictionary to serialize

        Returns:
            JSON-encoded bytes
        """
        try:
            return json.dumps(data, default=str).encode("utf-8")
        except Exception as e:
            logger.error(f"Failed to serialize message data: {str(e)}")
            raise

    def _create_message_attributes(
        self, document_id: str, metadata: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Create message attributes for Pub/Sub message.

        Args:
            document_id: Document ID
            metadata: Additional metadata

        Returns:
            Dictionary of message attributes
        """
        attributes = {
            "document_id": document_id,
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "api",
        }

        # Add metadata as attributes (Pub/Sub attributes must be strings)
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                attributes[key] = str(value)
            elif value is not None:
                attributes[key] = json.dumps(value)

        return attributes

    @retry.Retry(
        predicate=retry.if_exception_type(
            Exception
        ),  # Retry on any exception
        initial=1.0,  # Initial delay in seconds
        maximum=60.0,  # Maximum delay in seconds
        multiplier=2.0,  # Exponential backoff multiplier
        deadline=300.0,  # Maximum total time in seconds
    )
    def _publish_with_retry(
        self,
        data: bytes,
        attributes: Dict[str, str],
        ordering_key: Optional[str] = None,
    ) -> str:
        """
        Publish message with retry logic.

        Args:
            data: Message data as bytes
            attributes: Message attributes dictionary
            ordering_key: Optional ordering key for ordered delivery

        Returns:
            Message ID
        """
        try:
            # Publish message with data, attributes, and optional ordering key
            # Only include ordering_key if it's provided and not None
            publish_kwargs = {
                **attributes,
            }
            if ordering_key:
                publish_kwargs["ordering_key"] = ordering_key
            
            future = self.publisher.publish(
                self.topic_path,
                data,
                **publish_kwargs,
            )
            message_id = future.result(timeout=30.0)  # 30 second timeout
            logger.info(f"Published message with ID: {message_id}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to publish message after retries: {str(e)}")
            raise

    def publish_document_task(
        self,
        document_id: str,
        metadata: Dict[str, str],
        ordering_key: Optional[str] = None,
    ) -> str:
        """
        Publish a document processing task to Pub/Sub.

        Args:
            document_id: Document ID to process
            metadata: Additional metadata for the task
            ordering_key: Optional ordering key for ordered delivery (e.g., user_id)

        Returns:
            Published message ID

        Raises:
            RuntimeError: If publisher is not initialized
            Exception: If publishing fails after retries
        """
        if not self._publisher:
            self.initialize()

        try:
            # Create message payload
            message_data = {
                "document_id": document_id,
                "user_id": metadata.get("user_id", ""),
                "storage_path": metadata.get("storage_path", ""),
                "document_type": metadata.get("document_type", ""),
                "metadata": {
                    "filename": metadata.get("filename", ""),
                    "uploaded_at": metadata.get("uploaded_at", ""),
                    **{k: v for k, v in metadata.items() if k not in ["user_id", "storage_path", "document_type", "filename", "uploaded_at"]},
                },
            }

            # Serialize message data
            message_bytes = self._serialize_message(message_data)

            # Create message attributes
            message_attributes = self._create_message_attributes(document_id, metadata)

            # Publish with retry logic
            message_id = self._publish_with_retry(
                message_bytes, message_attributes, ordering_key=ordering_key
            )

            logger.info(
                f"Published document processing task for document: {document_id}, "
                f"message_id: {message_id}"
            )

            return message_id

        except Exception as e:
            logger.error(
                f"Failed to publish document task for document {document_id}: {str(e)}"
            )
            raise

    def publish_batch(
        self,
        tasks: List[Dict[str, Any]],
        ordering_key: Optional[str] = None,
    ) -> List[str]:
        """
        Publish multiple document processing tasks in a batch.

        Args:
            tasks: List of task dictionaries, each containing 'document_id' and 'metadata'
            ordering_key: Optional ordering key for ordered delivery

        Returns:
            List of published message IDs
        """
        if not self._publisher:
            self.initialize()

        message_ids = []

        try:
            for task in tasks:
                document_id = task.get("document_id")
                metadata = task.get("metadata", {})

                if not document_id:
                    logger.warning("Skipping task without document_id")
                    continue

                message_id = self.publish_document_task(
                    document_id, metadata, ordering_key=ordering_key
                )
                message_ids.append(message_id)

            logger.info(f"Published {len(message_ids)} document tasks in batch")
            return message_ids

        except Exception as e:
            logger.error(f"Failed to publish batch tasks: {str(e)}")
            raise

    def get_topic_info(self) -> Dict[str, str]:
        """
        Get information about the Pub/Sub topic.

        Returns:
            Dictionary with topic information
        """
        if not self._publisher:
            self.initialize()

        try:
            topic = self.publisher.get_topic(request={"topic": self.topic_path})
            return {
                "topic_name": settings.pubsub_topic_name,
                "topic_path": self.topic_path,
                "message_retention_duration": str(
                    topic.message_retention_duration
                ),
            }
        except Exception as e:
            logger.warning(f"Failed to get topic info: {str(e)}")
            return {
                "topic_name": settings.pubsub_topic_name,
                "topic_path": self.topic_path,
                "error": str(e),
            }


# Global service instance
pubsub_service = PubSubService()
