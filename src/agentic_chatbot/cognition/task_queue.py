"""Background task queue for non-blocking System 3 learning.

Implements a PostgreSQL-backed task queue with a single worker that:
- Polls for pending tasks
- Processes tasks with retry support
- Handles graceful shutdown
- Uses exponential backoff for retries

The worker runs as an asyncio task during application lifespan,
ensuring tasks complete reliably without blocking user responses.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Awaitable, Any

from agentic_chatbot.cognition.config import CognitionSettings, get_cognition_settings
from agentic_chatbot.cognition.models import LearningTask, TaskType, TaskStatus
from agentic_chatbot.utils.logging import get_logger

if TYPE_CHECKING:
    from agentic_chatbot.cognition.storage import CognitionStorage


logger = get_logger(__name__)


# Type alias for task handlers
TaskHandler = Callable[[dict[str, Any]], Awaitable[None]]


class CognitionTaskQueue:
    """
    PostgreSQL-backed task queue for background learning.

    Features:
    - Single worker design for simplicity
    - Atomic task claiming (FOR UPDATE SKIP LOCKED)
    - Retry with exponential backoff
    - Graceful shutdown (completes current task)
    - Task handler registration
    """

    def __init__(
        self,
        storage: CognitionStorage,
        settings: CognitionSettings | None = None,
    ):
        """
        Initialize task queue.

        Args:
            storage: CognitionStorage instance for database operations
            settings: Optional settings (uses defaults if not provided)
        """
        self.storage = storage
        self.settings = settings or get_cognition_settings()
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._handlers: dict[TaskType, TaskHandler] = {}

    def register_handler(self, task_type: TaskType, handler: TaskHandler) -> None:
        """
        Register a handler for a task type.

        Args:
            task_type: Type of task to handle
            handler: Async function that processes the task payload
        """
        self._handlers[task_type] = handler
        logger.debug(f"Registered handler for task type: {task_type.value}")

    async def enqueue(
        self,
        task_type: TaskType,
        payload: dict[str, Any],
        delay_seconds: int = 0,
    ) -> str:
        """
        Add a task to the queue.

        This is a fast, non-blocking operation that returns immediately.

        Args:
            task_type: Type of task
            payload: Task data
            delay_seconds: Optional delay before task becomes eligible

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        scheduled_for = datetime.utcnow() + timedelta(seconds=delay_seconds)

        task = LearningTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            scheduled_for=scheduled_for,
        )

        await self.storage.enqueue_task(task)
        logger.debug(f"Enqueued task {task_id} of type {task_type.value}")

        return task_id

    async def start_worker(self) -> None:
        """
        Start the background worker.

        Should be called during application startup.
        """
        if self._running:
            logger.warning("Worker already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Cognition task queue worker started")

    async def stop_worker(self) -> None:
        """
        Stop the worker gracefully.

        Waits for the current task to complete (with timeout).
        Should be called during application shutdown.
        """
        if not self._running:
            return

        logger.info("Stopping cognition task queue worker...")
        self._running = False

        if self._worker_task:
            try:
                # Wait for current task with timeout
                await asyncio.wait_for(
                    self._worker_task,
                    timeout=self.settings.task_worker_shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Worker shutdown timeout, cancelling...")
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass

        logger.info("Cognition task queue worker stopped")

    async def _worker_loop(self) -> None:
        """
        Main worker loop.

        Continuously polls for and processes tasks until shutdown.
        """
        logger.info("Worker loop started")

        while self._running:
            try:
                # Try to claim a task
                task = await self.storage.claim_next_task()

                if task:
                    await self._process_task(task)
                else:
                    # No pending tasks, wait before polling again
                    await asyncio.sleep(self.settings.task_poll_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                # Back off on errors to prevent tight error loops
                await asyncio.sleep(5.0)

        logger.info("Worker loop exited")

    async def _process_task(self, task: LearningTask) -> None:
        """
        Process a single task.

        Args:
            task: Task to process
        """
        logger.info(
            f"Processing task {task.task_id}",
            task_type=task.task_type.value,
            attempt=task.attempts,
        )

        try:
            # Get handler for task type
            handler = self._handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.task_type.value}")

            # Execute handler
            await handler(task.payload)

            # Mark as completed
            await self.storage.complete_task(task.task_id)
            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            logger.error(
                f"Task {task.task_id} failed: {e}",
                task_type=task.task_type.value,
                attempt=task.attempts,
                exc_info=True,
            )

            # Mark as failed with retry
            retry = task.attempts < task.max_attempts
            await self.storage.fail_task(task.task_id, str(e), retry=retry)

            if retry:
                backoff = 2 ** task.attempts
                logger.info(f"Task {task.task_id} scheduled for retry in {backoff}s")
            else:
                logger.warning(f"Task {task.task_id} max retries exceeded, marked as failed")

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    async def get_queue_stats(self) -> dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dict with counts by status
        """
        async with self.storage.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT status, COUNT(*) as count
                FROM cognition_tasks
                GROUP BY status
                """
            )
            return {row["status"]: row["count"] for row in rows}
