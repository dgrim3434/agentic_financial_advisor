from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List

from crewai import Agent, Task, Crew, LLM

from src.agents.models import UserProfile



# LLM helper

def _get_planner_llm() -> LLM:
    """
    LLM used for planning the ticker universe.

    Right now this is pointed at a local Ollama Llama 3 instance.
    If you want to switch to OpenAI later, you can change this to:

        return LLM(model="gpt-4.1-mini")

    and make sure OPENAI_API_KEY is set in your environment.
    """
    return LLM(
        model="ollama/llama3",
        base_url="http://localhost:11434",
    )



# Agent + Task builders

def _build_ticker_planner_agent() -> Agent:
    """
    Agent whose only job is to output JSON and nothing else.
    No chain-of-thought, no explanation, just a JSON object.
    """
    llm = _get_planner_llm()
    return Agent(
        role="Ticker Universe Planner",
        goal=(
            "Return STRICT JSON ONLY. Never add thoughts, explanations, reasoning, "
            "or any text outside the JSON object."
        ),
        backstory=(
            "You MUST always output valid JSON and nothing else. Do not explain your reasoning. "
            "Do not include 'Thought:', 'Action:', or any natural language text. "
            "Your entire response MUST be a single JSON object only."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def _build_ticker_planner_task(agent: Agent) -> Task:
    """
    Task that tells the planner exactly what JSON schema to follow.
    """
    return Task(
        description=(
            "!!! IMPORTANT: OUTPUT MUST BE STRICT JSON ONLY WITH NO EXTRA TEXT !!!\n\n"
            "Given the following inputs:\n"
            "{investor_profile_json}\n"
            "{user_watchlist}\n\n"
            "Generate a JSON object with this exact schema:\n"
            "{\n"
            '  "primary_tickers": [string],\n'
            '  "similar_tickers": [string],\n'
            '  "balancing_tickers": [string],\n'
            '  "other_tickers": [string]\n'
            "}\n\n"
            "RULES:\n"
            "- PRIMARY MUST include all user_watchlist tickers.\n"
            "- No markdown, no english sentences, no descriptors.\n"
            "- No prefixes like 'Thought:', 'Action:', 'Final Answer:'.\n"
            "- Output MUST start with '{' and end with '}'.\n"
            "- If you break JSON, the system fails.\n\n"
            "Return ONLY the JSON."
        ),
        expected_output="Valid JSON following the schema above.",
        agent=agent,
    )



# Public API

def run_ticker_planner(profile: UserProfile) -> Dict[str, List[str]]:
    """
    Run the planner LLM to decide the full ticker universe for research.

    Returns a dict of upper-cased tickers with the shape:

        {
          "primary_tickers":   [str],
          "similar_tickers":   [str],
          "balancing_tickers": [str],
          "other_tickers":     [str],
        }

    This is what `investment_strategy_crew.run_research_universe` consumes.
    """
    agent = _build_ticker_planner_agent()
    task = _build_ticker_planner_task(agent)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
        process="sequential",
    )

    watchlist_str = ",".join(profile.ticker_watchlist or [])

    inputs = {
        "investor_profile_json": json.dumps(asdict(profile), indent=2),
        "user_watchlist": watchlist_str,
    }

    # Run the crew with explicit inputs
    result = crew.kickoff(inputs=inputs)

    # CrewAI usually exposes final text on .output; fall back to str(result) if needed.
    raw_text = getattr(result, "output", None) or str(result)

    print("[DEBUG] Raw planner output:")
    print(raw_text)
    print("[DEBUG] --- end raw planner output ---\n")

    # We designed the prompt for JSON-only output, so we parse directly.
    plan = json.loads(raw_text)

    # Normalize into lists of upper-case strings
    def _norm_list(key: str) -> List[str]:
        values = plan.get(key, [])
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for v in values:
            s = str(v).strip().upper()
            if s:
                out.append(s)
        return out

    parsed_plan: Dict[str, List[str]] = {
        "primary_tickers": _norm_list("primary_tickers"),
        "similar_tickers": _norm_list("similar_tickers"),
        "balancing_tickers": _norm_list("balancing_tickers"),
        "other_tickers": _norm_list("other_tickers"),
    }

    print("[DEBUG] Parsed plan:")
    print(json.dumps(parsed_plan, indent=2))
    print()

    return parsed_plan



# Standalone debug runner

if __name__ == "__main__":
    """
    Quick manual test for the ticker planner.

    Run from the project root:

        python -m src.agents.ticker_planner
    """
    from src.agents.models import UserProfile

    profile = UserProfile(
        name="Debug User",
        age=22,
        risk_tolerance="medium",
        investment_horizon="10+ years",
        experience_level="beginner",
        preferred_sectors=["Technology", "AI"],
        investment_amount="10000",
        ticker_watchlist=["AAPL"],
        notes="Debug profile for ticker planner test.",
    )

    plan = run_ticker_planner(profile)
    print("[DEBUG] Final ticker plan:")
    print(json.dumps(plan, indent=2))
