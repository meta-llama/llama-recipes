"""
Context Layers — Give agents persistent identity + dynamic state.

4 layers, from most stable to most dynamic:
  Layer 1 (Identity): Who the agent IS. Never changes mid-session.
  Layer 2 (State): What's happening NOW. Changes every call.
  Layer 3 (Relevant): Knowledge retrieved for this specific task.
  Layer 4 (Archive): Everything stored, not loaded unless searched.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentContext:
    """Assembled context for an agent, built from 4 layers."""

    # Layer 1: Identity (static per session)
    name: str
    role: str
    instructions: str

    # Layer 2: State (dynamic per call)
    current_task: str = ""
    previous_output: str = ""

    # Layer 3: Relevant (retrieved per task)
    knowledge: list[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Assemble all layers into a system prompt."""
        parts = [
            f"# Identity\nYou are {self.name}, a {self.role}.\n{self.instructions}",
        ]
        if self.current_task:
            parts.append(f"\n# Current State\nTask: {self.current_task}")
        if self.previous_output:
            parts.append(f"Previous step output:\n{self.previous_output}")
        if self.knowledge:
            parts.append(
                "\n# Relevant Knowledge\n" + "\n".join(f"- {k}" for k in self.knowledge)
            )
        return "\n".join(parts)


# Pre-built agent identities
RESEARCHER = AgentContext(
    name="Researcher",
    role="research analyst",
    instructions=(
        "You analyze topics thoroughly and produce structured findings.\n"
        "Output JSON with: {summary, key_points[], risks[], opportunities[]}\n"
        "Be specific. Cite reasoning. Flag uncertainties."
    ),
)

BUILDER = AgentContext(
    name="Builder",
    role="implementation engineer",
    instructions=(
        "You build solutions based on research findings.\n"
        "Output working code or detailed implementation plan.\n"
        "Prioritize: correctness > speed > elegance."
    ),
)

REVIEWER = AgentContext(
    name="Reviewer",
    role="quality reviewer",
    instructions=(
        "You evaluate solutions against quality criteria.\n"
        "Output JSON with: {score (1-10), strengths[], issues[], verdict}\n"
        "Score >= 7 = PASS. Score < 7 = needs revision.\n"
        "Be constructive but honest. No rubber stamping."
    ),
)
