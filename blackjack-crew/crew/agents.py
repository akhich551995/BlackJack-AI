import os
from crewai import Agent
from crew.tools import GameActionTool
from typing import Any, Optional, List
from pydantic import BaseModel # Added this to ensure type resolution is clean
import os


def get_agents(game_tool: GameActionTool, ai_names: List[str], llm_instance: Optional[Any] = None) -> dict:
    """Create Agent instances for the provided AI player names.

    - `ai_names` is a list of strings like ['AI_1', 'AI_2'].
    - The first AI in the list is assigned the 'Strategist' role; remaining
      AIs are assigned the 'Cautious' role by default.
    - Human players are intentionally omitted from this agents dict and are
      handled synchronously by `main.py`.

    If `llm_instance` is provided it will be passed to each Agent as `llm`.
    Otherwise, Agents will be created using the model string from the
    CREWAI_MODEL environment variable.
    """

    model_string = None
    if llm_instance is None:
        model_string = os.getenv('CREWAI_MODEL')

    def _agent_llm_kwargs():
        if llm_instance is not None:
            return {'llm': llm_instance}
        elif model_string:
            return {'model': model_string}
        else:
            return {}

    agents = {}

    # Create AI agents according to the provided names and assign roles
    for idx, name in enumerate(ai_names):
        if idx == 0:
            role = 'AI Blackjack Strategist'
            goal = 'Determine the mathematically optimal move (Hit or Stand) based on the Basic Strategy chart for Blackjack.'
            backstory = "You are a genius AI player who always chooses the mathematically optimal action to minimize the house edge."
        else:
            role = 'AI Cautious Player'
            goal = 'Play cautiously: Stand on any score of 15 or higher. Hit only if your score is 14 or lower.'
            backstory = "You are an AI who prioritizes safety over aggression, always trying to avoid a bust."

        agents[name] = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=[game_tool],
            verbose=True,
            allow_delegation=False,
            **_agent_llm_kwargs()
        )

    return agents