"""
Safety Guard — Validate LLM output, track costs, enable rollback.

Simplified version of Penguin Guard (github.com/PenguinAlleyApps/penguin-guard).
"""
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("safety_guard")


class SafetyGuard:
    """Wraps LLM calls with validation, cost tracking, and rollback."""

    def __init__(self, max_cost_usd: float = 10.0, cost_per_1m_tokens: float = 0.20):
        self.max_cost_usd = max_cost_usd
        self.cost_per_1m_tokens = cost_per_1m_tokens
        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0
        self._last_good: dict[str, Any] = {}

    def validate_json(self, text: str, required_fields: list[str] | None = None) -> tuple[bool, Any]:
        """Validate that output is parseable JSON with required fields."""
        # Try direct parse
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try extracting from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return False, "JSON parse failed (even from code block)"
            else:
                return False, "Not valid JSON"

        if required_fields:
            missing = [f for f in required_fields if f not in data]
            if missing:
                return False, f"Missing fields: {missing}"

        return True, data

    def track_cost(self, text: str) -> bool:
        """Track token cost. Returns False if budget exceeded."""
        tokens = len(text) // 4  # rough estimate
        cost = (tokens / 1_000_000) * self.cost_per_1m_tokens
        self.total_cost += cost
        self.total_tokens += tokens
        self.call_count += 1

        if self.total_cost >= self.max_cost_usd:
            logger.warning(f"Budget exceeded: ${self.total_cost:.4f} >= ${self.max_cost_usd}")
            return False
        return True

    def save_good(self, agent_name: str, output: Any):
        """Save a known-good output for rollback."""
        self._last_good[agent_name] = output

    def rollback(self, agent_name: str) -> Any | None:
        """Get last known-good output for an agent."""
        return self._last_good.get(agent_name)

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "budget_remaining": round(self.max_cost_usd - self.total_cost, 4),
        }
