"""Collect result node."""

from typing import Any

from ..base import AsyncBaseNode
from ...utils.logging import get_logger

logger = get_logger(__name__)


class CollectResultNode(AsyncBaseNode):
    """
    Collects and stores operator output.
    """

    name = "collect_result"

    async def execute(self, shared: dict[str, Any]) -> str:
        latest_result = shared.get("latest_result")
        if latest_result:
            results = shared.get("collected_results", [])
            results.append(latest_result)
            shared["collected_results"] = results

        return "observe"
