"""
Agent Cards — Capability discovery for multi-agent routing.

Instead of hardcoding "send this to Agent X", agents publish
their capabilities as JSON cards. The orchestrator searches
cards to find the right agent for each task.
"""
from dataclasses import dataclass, field


@dataclass
class AgentCard:
    """JSON-serializable capability descriptor for an agent."""
    name: str
    capabilities: list[str]
    inputs: list[str]
    outputs: list[str]

    def matches(self, query: str) -> bool:
        """Check if this agent matches a capability query."""
        q = query.lower()
        return any(q in cap.lower() for cap in self.capabilities)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capabilities": self.capabilities,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


class CardDirectory:
    """Registry of all agent cards. Agents find each other here."""

    def __init__(self):
        self._cards: dict[str, AgentCard] = {}

    def register(self, card: AgentCard):
        self._cards[card.name] = card

    def find(self, capability: str) -> list[AgentCard]:
        """Find agents that match a capability query."""
        return [c for c in self._cards.values() if c.matches(capability)]

    def get(self, name: str) -> AgentCard | None:
        return self._cards.get(name)

    def list_all(self) -> list[AgentCard]:
        return list(self._cards.values())


# Pre-built directory for the 3-agent pipeline
def build_default_directory() -> CardDirectory:
    directory = CardDirectory()

    directory.register(AgentCard(
        name="Researcher",
        capabilities=["research", "analysis", "market_study", "competitive_intel"],
        inputs=["topic", "question", "domain"],
        outputs=["structured_findings_json"],
    ))

    directory.register(AgentCard(
        name="Builder",
        capabilities=["build", "implement", "code", "create", "develop"],
        inputs=["research_findings", "spec", "requirements"],
        outputs=["code", "implementation_plan"],
    ))

    directory.register(AgentCard(
        name="Reviewer",
        capabilities=["review", "evaluate", "qa", "quality_check"],
        inputs=["solution", "criteria", "research_context"],
        outputs=["evaluation_json", "score", "verdict"],
    ))

    return directory
