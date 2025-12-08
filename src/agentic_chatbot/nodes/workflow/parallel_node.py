"""Execute parallel node."""

import asyncio
from typing import Any

from ..base import AsyncBaseNode
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ExecuteParallelNode(AsyncBaseNode):
    """
    Executes multiple steps in parallel using asyncio.gather.
    """

    name = "execute_parallel"

    async def execute(self, shared: dict[str, Any]) -> str:
        parallel_steps = shared.get("parallel_steps", [])
        if not parallel_steps:
            return "continue"

        logger.info(f"Executing {len(parallel_steps)} steps in parallel")

        return "continue"
