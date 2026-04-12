"""
Multi-Agent Orchestrator — 3-agent pipeline with phase gates.

Usage:
    python orchestrator.py "Build a CLI tool that converts CSV to JSON"

Requires:
    LLAMA_BASE_URL (default: http://localhost:11434/v1 for Ollama)
    LLAMA_MODEL (default: llama3.2)
"""
import json
import os
import sys
from typing import Optional

from context_layers import AgentContext, RESEARCHER, BUILDER, REVIEWER
from agent_cards import build_default_directory
from safety_guard import SafetyGuard

# --- LLM Client (OpenAI-compatible, works with Ollama / vLLM / llama.cpp) ---

def call_llm(system: str, user: str, model: Optional[str] = None) -> str:
    """Call an OpenAI-compatible LLM endpoint."""
    import requests

    base_url = os.getenv("LLAMA_BASE_URL", "http://localhost:11434/v1")
    model_name = model or os.getenv("LLAMA_MODEL", "llama3.2")

    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# --- Pipeline ---

def run_pipeline(task: str, verbose: bool = True):
    """Run the 3-agent pipeline: Research → Build → Review."""

    directory = build_default_directory()
    guard = SafetyGuard(max_cost_usd=5.0)

    # === PHASE 1: RESEARCH ===
    if verbose:
        print("\n=== Phase 1: Research ===")

    researcher_card = directory.find("research")[0]
    researcher = RESEARCHER
    researcher.current_task = task

    research_output = call_llm(
        system=researcher.to_system_prompt(),
        user=f"Research this task and produce structured findings as JSON:\n\n{task}",
    )

    # Safety check: validate JSON output
    valid, research_data = guard.validate_json(
        research_output, required_fields=["summary", "key_points"]
    )
    if not valid:
        print(f"  WARNING: Research output not valid JSON ({research_data})")
        print(f"  Using raw text as findings")
        research_data = {"summary": research_output, "key_points": []}
    else:
        guard.save_good("Researcher", research_data)

    guard.track_cost(research_output)

    if verbose:
        summary = research_data.get("summary", research_output[:200])
        print(f"  Findings: {summary[:150]}...")

    # === PHASE 2: BUILD ===
    if verbose:
        print("\n=== Phase 2: Build ===")

    builder_card = directory.find("build")[0]
    builder = BUILDER
    builder.current_task = task
    builder.previous_output = json.dumps(research_data) if isinstance(research_data, dict) else str(research_data)

    build_output = call_llm(
        system=builder.to_system_prompt(),
        user=(
            f"Based on these research findings, build a solution:\n\n"
            f"Research: {builder.previous_output[:1500]}\n\n"
            f"Original task: {task}"
        ),
    )

    guard.track_cost(build_output)
    guard.save_good("Builder", build_output)

    if verbose:
        print(f"  Solution: {build_output[:150]}...")

    # === PHASE 3: REVIEW (phase gate) ===
    if verbose:
        print("\n=== Phase 3: Review ===")

    reviewer_card = directory.find("review")[0]
    reviewer = REVIEWER
    reviewer.current_task = f"Evaluate solution for: {task}"
    reviewer.previous_output = build_output
    reviewer.knowledge = [
        "Score >= 7 means PASS. Score < 7 means needs revision.",
        f"Research context: {json.dumps(research_data)[:500] if isinstance(research_data, dict) else str(research_data)[:500]}",
    ]

    review_output = call_llm(
        system=reviewer.to_system_prompt(),
        user=(
            f"Evaluate this solution:\n\n{build_output[:2000]}\n\n"
            f"Criteria: correctness, completeness, code quality, production-readiness.\n"
            f"Output JSON: {{score, strengths[], issues[], verdict}}"
        ),
    )

    guard.track_cost(review_output)

    valid, review_data = guard.validate_json(
        review_output, required_fields=["score", "verdict"]
    )

    if valid:
        score = review_data.get("score", 0)
        verdict = review_data.get("verdict", "unknown")
        guard.save_good("Reviewer", review_data)
    else:
        score = 0
        verdict = "parse_failed"
        print(f"  WARNING: Review not valid JSON ({review_data})")

    # === PHASE GATE ===
    passed = score >= 7

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Score: {score}/10 — {'PASS' if passed else 'NEEDS REVISION'}")
        print(f"  Verdict: {verdict}")
        if valid and "issues" in review_data:
            for issue in review_data["issues"]:
                print(f"  Issue: {issue}")
        print(f"\n  Cost: {json.dumps(guard.summary())}")

    return {
        "task": task,
        "research": research_data,
        "solution": build_output,
        "review": review_data if valid else review_output,
        "score": score,
        "passed": passed,
        "cost": guard.summary(),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py \"Your task description\"")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    result = run_pipeline(task)

    if not result["passed"]:
        print("\nPipeline result: NEEDS REVISION")
        print("In production, this would loop back to Builder with review feedback.")
    else:
        print("\nPipeline result: PASSED")
