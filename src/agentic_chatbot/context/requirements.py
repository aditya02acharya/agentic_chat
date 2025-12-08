"""Context requirements DSL parsing."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ContextRequirement:
    """Parsed context requirement."""

    type: str
    params: dict[str, Any]


def parse_requirement(req: str) -> ContextRequirement:
    """
    Parse a context requirement string.

    Examples:
        "query" -> ContextRequirement(type="query", params={})
        "tools.schema(web_search)" -> ContextRequirement(type="tools.schema", params={"tool": "web_search"})
        "results.last(3)" -> ContextRequirement(type="results.last", params={"count": 3})
    """
    if "(" not in req:
        return ContextRequirement(type=req, params={})

    paren_start = req.index("(")
    paren_end = req.rindex(")")
    req_type = req[:paren_start]
    param_str = req[paren_start + 1 : paren_end]

    params = {}
    if param_str:
        if param_str.isdigit():
            params["value"] = int(param_str)
        else:
            params["value"] = param_str

    return ContextRequirement(type=req_type, params=params)


def validate_requirements(requirements: list[str]) -> list[str]:
    """
    Validate context requirements.

    Returns list of validation errors (empty if valid).
    """
    errors = []
    valid_types = {
        "query",
        "conversation_history",
        "previous_results",
        "tools.schema",
        "action_history",
    }

    for req in requirements:
        parsed = parse_requirement(req)
        if parsed.type not in valid_types:
            errors.append(f"Unknown requirement type: {parsed.type}")

    return errors
