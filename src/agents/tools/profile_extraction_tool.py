import json
from typing import Dict, Any
from crewai.tools import tool
import ollama

@tool("extract_investor_profile")
def extract_investor_profile(text: str) -> Dict[str, Any]:
    """
    Extract investor profile from a user description.
    
    the profile sound include:
    - Name
    - Age
    - Risk_tolerance (low, medium, high)
    - investment_horizon (short, medium, long)
    - experience_level (begineer, intermediate, advanced)
    - preferred_sectors (list of strings)
    - ticker_watchlist (list of strings)
    - constraints
    """
    
    prompt = f"""
    You are an expert system for extracting structured investor profiles
    from natural language descriptions. Convert the following text into a JSON object:

    \"\"\"{text}\"\"\"

    Your JSON MUST use this schema:

    {{
    "name": string or null,
    "age": integer or null,
    "risk_tolerance": "low" | "medium" | "high" | null,
  "investment_horizon": "short" | "medium" | "long" | null,
    "experience_level": "beginner" | "intermediate" | "advanced" | null,
    "preferred_sectors": [string],
    "ticker_watchlist": [string],
    "constraints": string or null
    }}

    Important rules:
    - Do NOT hallucinate information.
    - If something is not explicitly stated, set the value to null.
    - If you infer sectors or tickers, they must be clearly mentioned.
    - Output ONLY valid JSON. No commentary.
    """
    
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "Extract structured data only."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0},
    )
    
    content = response["message"]["content"]
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(content[start : end + 1])
        else:
            raise ValueError("Model output could not be parsed as JSON:\n" + content)
    
    return data

