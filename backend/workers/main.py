"""
Pub/Sub worker for document processing.

This worker subscribes to Pub/Sub messages and processes documents
asynchronously through the complete pipeline: parsing, chunking, embedding, indexing.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from aiohttp import web
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.message import Message

from config import settings
from services import storage_service
from services.embedding_service import embedding_service
from workers.document_processor import document_processor

logger = logging.getLogger(__name__)

# Global state for graceful shutdown
shutdown_event = asyncio.Event()
executor = ThreadPoolExecutor(max_workers=10)
_shutdown_count = 0  # Track number of shutdown signals


class DocumentProcessingWorker:
    """Worker for processing documents from Pub/Sub messages."""

    def __init__(self):
        """Initialize document processing worker."""
        self.subscriber: Optional[pubsub_v1.SubscriberClient] = None
        self.subscription_path: Optional[str] = None
        self._running = False
        self._processing_tasks = set()  # Track async tasks
        self._processing_futures = set()  # Track thread pool futures
        self._processing_coroutines = set()  # Track coroutine futures from run_coroutine_threadsafe
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def initialize(self):
        """Initialize Pub/Sub subscriber and services."""
        try:
            # Initialize Pub/Sub subscriber
            self.subscriber = pubsub_v1.SubscriberClient()
            # Use subscription name based on topic name
            subscription_name = f"{settings.pubsub_topic_name}-sub"
            self.subscription_path = self.subscriber.subscription_path(
                settings.gcp_project_id,
                subscription_name,
            )

            # Initialize services
            storage_service.initialize()
            
            # Initialize embedding service (non-blocking - will fail gracefully if Vertex AI not configured)
            try:
                embedding_service.initialize()
            except Exception as e:
                logger.warning(
                    f"Embedding service initialization failed: {str(e)}. "
                    "Worker will continue but embedding generation will fail. "
                    "Enable Vertex AI API to fix this."
                )

            # Initialize Pinecone service (non-blocking - will fall back to Firestore if fails)
            from services.pinecone_service import pinecone_service
            try:
                if not pinecone_service.is_initialized:
                    pinecone_service.initialize()
                    logger.info("Pinecone service initialized for worker")
                else:
                    logger.info("Pinecone service already initialized")
            except Exception as e:
                logger.warning(
                    f"Pinecone service initialization failed: {str(e)}. "
                    "Worker will continue but embeddings will not be stored in vector database. "
                    "Check PINECONE_API_KEY and PINECONE_ENVIRONMENT configuration.",
                    exc_info=True
                )

            logger.info(
                f"Worker initialized for subscription: {self.subscription_path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize worker: {str(e)}")
            raise

    async def process_document(self, document_id: str, message_data: dict):
        """
        Process a document through the complete pipeline.

        Args:
            document_id: Document ID to process
            message_data: Message data from Pub/Sub
        """
        # Use document processor for processing
        await document_processor.process_document(
            document_id=document_id,
            message_data=message_data,
        )

    def callback(self, message: Message):
        """
        Callback function for Pub/Sub messages.

        This is a synchronous callback that spawns async tasks for processing.

        Args:
            message: Pub/Sub message
        """
        try:
            # Parse message data
            message_data = json.loads(message.data.decode("utf-8"))
            document_id = message_data.get("document_id")

            if not document_id:
                logger.error("Message missing document_id, nacking message")
                message.nack()
                return

            logger.info(f"Received message for document: {document_id}")

            # Schedule async processing using the event loop
            if self._event_loop and self._event_loop.is_running():
                # Use thread-safe scheduling
                future = asyncio.run_coroutine_threadsafe(
                    self._process_with_ack(message, document_id, message_data),
                    self._event_loop,
                )
                # Track the future
                self._processing_futures.add(future)
                self._processing_coroutines.add(future)
                future.add_done_callback(self._processing_futures.discard)
                future.add_done_callback(self._processing_coroutines.discard)
            else:
                # Fallback: process synchronously in thread pool
                logger.warning("Event loop not available, processing in thread pool")
                thread_future = executor.submit(
                    asyncio.run,
                    self._process_with_ack(message, document_id, message_data),
                )
                self._processing_futures.add(thread_future)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message data: {str(e)}")
            message.nack()  # Nack to retry
        except Exception as e:
            logger.error(f"Error in message callback: {str(e)}", exc_info=True)
            message.nack()  # Nack to retry

    async def _process_with_ack(
        self, message: Message, document_id: str, message_data: dict
    ):
        """
        Process document and acknowledge message.

        Args:
            message: Pub/Sub message
            document_id: Document ID
            message_data: Message data
        """
        try:
            await self.process_document(document_id, message_data)
            # Acknowledge message after successful processing
            message.ack()
            logger.info(f"Message acknowledged for document: {document_id}")
        except Exception as e:
            logger.error(
                f"Failed to process document {document_id}: {str(e)}", exc_info=True
            )
            # Nack message to retry (Pub/Sub will handle retries and DLQ)
            message.nack()

    async def start(self):
        """Start the worker and begin processing messages."""
        if not self.subscriber:
            self.initialize()

        self._running = True
        self._event_loop = asyncio.get_event_loop()

        logger.info("Starting document processing worker...")

        # Create streaming pull future
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path, callback=self.callback
        )

        logger.info(
            f"Listening for messages on {self.subscription_path}...\n"
            "Press Ctrl+C to stop."
        )

        try:
            # Wait for shutdown signal
            await shutdown_event.wait()

            logger.info("Shutdown signal received, stopping worker...")

            # Cancel streaming pull
            streaming_pull_future.cancel()

            # Wait for streaming pull to finish
            try:
                streaming_pull_future.result(timeout=10)
            except Exception:
                pass

            # Wait for all processing tasks to complete (with shorter timeout for graceful shutdown)
            all_futures = list(self._processing_futures)
            if all_futures:
                logger.info(
                    f"Waiting up to 10 seconds for {len(all_futures)} processing task(s) to complete..."
                )
                import time
                start_time = time.time()
                max_wait_time = 10  # Maximum 10 seconds for graceful shutdown
                
                # Check which futures are done
                done_futures = [f for f in all_futures if hasattr(f, "done") and f.done()]
                if done_futures:
                    logger.info(f"{len(done_futures)} task(s) already completed")
                
                try:
                    # Wait for all futures to complete with overall timeout
                    for future in all_futures:
                        elapsed = time.time() - start_time
                        remaining_time = max(0, max_wait_time - elapsed)
                        
                        if remaining_time <= 0:
                            logger.warning(
                                "Shutdown timeout reached (10s). Exiting. "
                                "Some tasks may still be running and will be nacked by Pub/Sub."
                            )
                            break
                        
                        try:
                            if hasattr(future, "result"):
                                # Use shorter timeout per future
                                future.result(timeout=min(remaining_time, 5))
                                logger.debug("Task completed successfully")
                        except Exception as e:
                            # TimeoutError or other exceptions - task may still be running
                            if "timeout" in str(e).lower() or isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                                logger.warning(
                                    f"Task timeout after {elapsed:.1f}s. "
                                    "Task may still be running. Pub/Sub will redeliver if needed."
                                )
                            else:
                                logger.warning(f"Task completed with exception: {str(e)}")
                except Exception as e:
                    logger.warning(
                        f"Error waiting for processing tasks: {str(e)}. "
                        "Exiting. Pub/Sub will handle message redelivery if needed."
                    )

            logger.info("Worker stopped gracefully")

        except Exception as e:
            logger.error(f"Error in worker: {str(e)}", exc_info=True)
            raise
        finally:
            self._running = False
            if self.subscriber:
                self.subscriber.close()

    def stop(self):
        """Stop the worker gracefully."""
        logger.info("Stopping worker...")
        shutdown_event.set()


def setup_signal_handlers(worker: DocumentProcessingWorker):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        global _shutdown_count
        _shutdown_count += 1
        
        if _shutdown_count == 1:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            logger.info("Press Ctrl+C again to force exit immediately.")
            worker.stop()
        elif _shutdown_count >= 2:
            logger.warning("Force exit requested. Terminating immediately...")
            import sys
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def health_check_handler(request):
    """HTTP health check endpoint for Cloud Run."""
    return web.Response(text="OK", status=200)


async def run_health_server():
    """Run HTTP health check server for Cloud Run."""
    app = web.Application()
    app.router.add_get("/health", health_check_handler)
    app.router.add_get("/", health_check_handler)  # Root path also works
    
    # Get port from environment or default to 8080 (Cloud Run default)
    port = int(os.environ.get("PORT", "8080"))
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Health check server started on port {port}")
    
    try:
        # Keep server running until shutdown event is set
        await shutdown_event.wait()
    finally:
        await runner.cleanup()
        logger.info("Health check server stopped")


async def main():
    """Main entry point for the worker."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and initialize worker
    worker = DocumentProcessingWorker()
    worker.initialize()

    # Setup signal handlers
    setup_signal_handlers(worker)

    try:
        # Start worker and health server concurrently
        await asyncio.gather(
            worker.start(),
            run_health_server(),
            return_exceptions=True,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        worker.stop()
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
