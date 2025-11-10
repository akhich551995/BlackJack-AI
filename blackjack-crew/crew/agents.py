import os
from crewai import Agent
from crew.tools import GameActionTool
from typing import Any, Optional, List


def get_agents(game_tool: GameActionTool, ai_names: List[str], llm_instance: Optional[Any] = None) -> dict:
    """Create AI Agent objects for the given names.

    First AI gets a 'Strategist' role; remaining AIs are 'Cautious'.
    Passes either an llm instance or a model string to Agent.
    """

    model_string = None
    if llm_instance is None:
        model_string = os.getenv('CREWAI_MODEL')

    def _agent_llm_kwargs():
        if llm_instance is not None:
            return {'llm': llm_instance}
        if model_string:
            return {'model': model_string}
        return {}

    agents = {}
    for idx, name in enumerate(ai_names):
        if idx == 0:
            role = 'AI Blackjack Strategist'
            goal = 'Select the optimal Hit/Stand based on basic strategy.'
            backstory = 'Plays to minimize house edge.'
        else:
            role = 'AI Cautious Player'
            goal = 'Stand on 15+; hit on 14 or less.'
            backstory = 'Prioritizes avoiding busts.'

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