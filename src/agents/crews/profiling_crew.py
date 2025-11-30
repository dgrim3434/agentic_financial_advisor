from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Dict, Tuple, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agents.models import UserProfile


# LLM setup (OpenAI gpt-4o-mini)

# Expects OPENAI_API_KEY in the environment, e.g.:

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    temperature=0,
)


# Helper: figure out which profile fields are still missing


# Core fields we need before we can build a portfolio
REQUIRED_FIELDS = ["risk_tolerance", "investment_horizon", "investment_amount"]


def compute_missing_fields(profile: UserProfile) -> List[str]:
    """
    Figure out which important fields are still missing for the user.

    We always check risk_tolerance, investment_horizon, and investment_amount,
    and we also surface ticker_watchlist status so the LLM knows whether
    it should ask about specific companies.
    """
    missing: List[str] = []

    if not profile.risk_tolerance:
        missing.append("risk_tolerance")

    if not profile.investment_horizon:
        missing.append("investment_horizon")

    if not profile.investment_amount:
        missing.append("investment_amount")

    # Tickers are optional, but we still expose this so the model knows
    # whether it should ask about specific companies or just choose for them.
    if not profile.ticker_watchlist:
        missing.append("ticker_watchlist (companies/tickers of interest)")

    return missing



# Pydantic schema for structured LLM output


class ProfileModel(BaseModel):
    name: Optional[str] = Field(default=None)
    age: Optional[int] = Field(default=None)

    risk_tolerance: Optional[str] = Field(
        default=None,
        description='One of: "low", "medium", "high".',
    )
    investment_horizon: Optional[str] = Field(
        default=None,
        description='One of: "short", "medium", "long".',
    )
    experience_level: Optional[str] = Field(
        default=None,
        description='One of: "beginner", "intermediate", "advanced".',
    )

    preferred_sectors: List[str] = Field(
        default_factory=list,
        description="List of sectors or themes, e.g. ['tech', 'healthcare'].",
    )
    ticker_watchlist: List[str] = Field(
        default_factory=list,
        description="List of stock tickers, e.g. ['AAPL', 'MSFT'].",
    )
    constraints: Optional[str] = Field(
        default=None,
        description="Any ethical / personal constraints (e.g. no tobacco).",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any extra free-form notes.",
    )
    investment_amount: Optional[str] = Field(
        default=None,
        description="How much the user wants to invest (e.g. '10000', 'around $5k').",
    )


class ProfilingTurnOutput(BaseModel):
    assistant_reply: str = Field(
        description="What the assistant says back to the user in natural language."
    )
    updated_profile: ProfileModel
    done: bool = Field(
        description=(
            "True if risk_tolerance, investment_horizon, and investment_amount "
            "are all filled and the profile is ready for portfolio construction."
        )
    )



# System prompt for the profiling LLM


SYSTEM_PROMPT = """
You are an AI investment onboarding assistant.

Your job is to build a structured UserProfile for an individual investor.
You must interact conversationally and ask follow-up questions when needed.

The UserProfile fields are:

- name: string or null
- age: integer or null
- risk_tolerance: "low" | "medium" | "high" | null
- investment_horizon: "short" | "medium" | "long" | null
- experience_level: "beginner" | "intermediate" | "advanced" | null
- preferred_sectors: list of strings (e.g. ["tech", "healthcare"])
- ticker_watchlist: list of stock tickers (e.g. ["AAPL", "MSFT"])
- constraints: string or null (e.g. "no tobacco or weapons")
- notes: string or null (freeform)
- investment_amount: string or number describing how much they want to invest
  (examples: "10000", "around 5k", "about $20,000")

MINIMUM REQUIREMENTS before we can build a portfolio:
- risk_tolerance
- investment_horizon
- investment_amount

You will ALWAYS receive:
- The CURRENT_PROFILE_JSON (your current best guess of the profile),
- A list of MISSING_FIELDS that are most important still to fill,
- The conversation so far,
- The latest USER message.

LOGIC RULES:

1. First, update the profile using any new info the user gave.
   Overwrite fields if the user corrects themselves.
   Append to lists (preferred_sectors, ticker_watchlist) if new items appear.

2. Decide if the profile is COMPLETE:
   - complete = risk_tolerance, investment_horizon, and investment_amount are all non-null.

3. If NOT complete:
   - Ask EXACTLY ONE short follow-up question.
   - Target the MOST important missing field from the MISSING_FIELDS list (in order).
   - Be concise and specific, no long explanations.

4. TICKER RULE (important):
   - If ticker_watchlist is in MISSING_FIELDS but the user clearly said they **do not**
     have any specific tickers/companies in mind, then:
       - keep ticker_watchlist as an empty list,
       - DO NOT ask about tickers again,
       - This is fine; the portfolio can be fully system-chosen.
   - Otherwise, if ticker_watchlist is missing and the user has not answered that yet,
     ask something like:
       "Do you have any specific companies or tickers in mind, or should I choose everything for you?"

5. When the profile IS complete:
   - First, update the profile with any new information from this turn.
   - Your assistant_reply should be a short confirmation AND an invitation, e.g.:
       "Great, I now have enough information to design a portfolio for you. Before we move on, is there anything else you'd like to add or any constraints I should know about?"
   - Do NOT design the portfolio yourself. That will be done by another system.
   - Set done = true (we are ready), but you must still incorporate any future
     user messages that add constraints, notes, or clarifications.


6. OUTPUT FORMAT:
   You MUST produce a JSON object with keys:
     - assistant_reply
     - updated_profile
     - done

   This JSON is parsed directly by the system. Do not include any extra keys or text.
"""



# Core step function (one turn of the chat)


def profiling_step(
    user_message: str,
    profile: UserProfile,
    history: List[Dict[str, str]],
) -> Tuple[str, UserProfile, bool, List[Dict[str, str]]]:
    """
    Run one turn of the onboarding conversation.

    Args:
        user_message: latest user input.
        profile: current UserProfile snapshot.
        history: list of {"role": "user"|"assistant", "content": "..."} messages.

    Returns:
        assistant_reply: what we show back to the user.
        new_profile: updated UserProfile with any new info applied.
        done: True if we have enough info to build a portfolio.
        new_history: history including this latest turn.
    """
    # Figure out what we're still missing before this turn
    missing_fields = compute_missing_fields(profile)

    # Flatten chat history into a simple text transcript for the model
    if history:
        history_lines = [
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ]
        history_block = "\n".join(history_lines)
    else:
        history_block = "None yet."

    # Build a single prompt string that includes the system prompt,
    # current profile JSON, missing fields, conversation, and latest message.
    prompt = f"""
{SYSTEM_PROMPT}

CURRENT_PROFILE_JSON:
{json.dumps(asdict(profile), indent=2)}

MISSING_FIELDS:
{json.dumps(missing_fields)}

CONVERSATION_HISTORY:
{history_block}

LATEST_USER_MESSAGE:
{user_message}
"""

    # Ask OpenAI for a structured ProfilingTurnOutput (Pydantic)
    structured_llm = llm.with_structured_output(ProfilingTurnOutput)
    result: ProfilingTurnOutput = structured_llm.invoke(prompt)

    # Convert ProfileModel -> our UserProfile dataclass
    updated = result.updated_profile
    new_profile = UserProfile(
        name=updated.name,
        age=updated.age,
        risk_tolerance=updated.risk_tolerance,
        investment_horizon=updated.investment_horizon,
        experience_level=updated.experience_level,
        preferred_sectors=updated.preferred_sectors or [],
        ticker_watchlist=updated.ticker_watchlist or [],
        constraints=updated.constraints,
        notes=updated.notes,
        investment_amount=updated.investment_amount,
    )

    # Append this turn to the conversation history
    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": result.assistant_reply},
    ]

    return result.assistant_reply, new_profile, result.done, new_history



# Convenience helper: one-shot profiling from a single blurb


def run_profiling(initial_description: str) -> UserProfile:
    """
    Quick helper for testing without a UI.

    Given one free-form description, we call profiling_step once
    (and at most a few times) and return the best profile we have.

    In the Streamlit app we drive profiling_step turn-by-turn instead.
    """
    profile = UserProfile()
    history: List[Dict[str, str]] = []
    user_message = initial_description

    for _ in range(5):  # small safety cap
        reply, profile, done, history = profiling_step(
            user_message=user_message,
            profile=profile,
            history=history,
        )
        # For this helper we don't simulate follow-up user replies,
        # so after the first turn we just return the best profile so far.
        break

    return profile


if __name__ == "__main__":
    example = """
    I'm Dylan, 22 years old, investing for long-term growth.
    I like AI and tech companies, and I'm currently watching AAPL, MSFT and NVDA.
    I'm okay with medium risk but nothing extremely volatile.
    I want to invest around $15,000 to start.
    """

    profile = run_profiling(example)
    print("Final profile from one-shot helper:\n", profile)
