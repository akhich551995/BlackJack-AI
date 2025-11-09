import os
from crewai import Agent
from crew.tools import GameActionTool
from typing import Any, Optional
from pydantic import BaseModel # Added this to ensure type resolution is clean
import os


def get_agents(game_tool: GameActionTool, llm_instance: Optional[Any] = None) -> dict:
    """Defines and returns all agents required for the Blackjack game.

    If an explicit `llm_instance` (an instantiated LLM object) is provided, it will
    be passed to Agents via the `llm` parameter. Otherwise, Agents will be
    constructed using the model string from the `CREWAI_MODEL` env var by passing
    `model=os.getenv('CREWAI_MODEL')` to the Agent constructor.
    """

    # If no LLM object was provided, use the CREWAI_MODEL env var so CrewAI
    # will construct the provider according to CREWAI_LLM_PROVIDER.
    model_string = None
    if llm_instance is None:
        model_string = os.getenv('CREWAI_MODEL')

    # Define Agents
    # Helper to pick how to pass LLM/model into Agent
    def _agent_llm_kwargs():
        if llm_instance is not None:
            return {'llm': llm_instance}
        elif model_string:
            return {'model': model_string}
        else:
            return {}

    dealer_agent = Agent(
        role='Blackjack Dealer',
        goal='Strictly follow house rules: manage the game flow, deal cards, and execute the Dealer’s turn (hit until 17 or more, then stand).',
        backstory='You are the professional casino Dealer. Your actions are procedural and fixed by the game rules. You only act when all players are done.',
        tools=[game_tool],
        verbose=True,
        allow_delegation=False,
        **_agent_llm_kwargs()
    )

    human_agent = Agent(
        role='Human Player Proxy',
        goal='Wait for the human user to provide a decision (Hit or Stand) and execute that choice using the Game Action Tool.',
        backstory='You are the direct input mechanism for the human user. Your function is to take the human\'s decision and pass it directly to the Game Action Tool. You do not use an LLM for decision making.',
        tools=[game_tool],
        # llm=None correctly forces human input
        llm=None,
        verbose=True,
        allow_delegation=False
    )

    ai_strategist_agent = Agent(
        role='AI Blackjack Strategist',
        goal='Determine the mathematically optimal move (Hit or Stand) based on the Basic Strategy chart for Blackjack.',
        backstory="You are a genius AI player who always chooses the mathematically optimal action to minimize the house edge.",
        tools=[game_tool],
        verbose=True,
        allow_delegation=False,
        **_agent_llm_kwargs()
    )

    ai_cautious_agent = Agent(
        role='AI Cautious Player',
        goal='Play cautiously: Stand on any score of 15 or higher. Hit only if your score is 14 or lower.',
        backstory="You are an AI who prioritizes safety over aggression, always trying to avoid a bust.",
        tools=[game_tool],
        verbose=True,
        allow_delegation=False,
        **_agent_llm_kwargs()
    )

    return {
        'Dealer': dealer_agent,
        'Human': human_agent,
        'AI_Strategist': ai_strategist_agent,
        'AI_Cautious': ai_cautious_agent
    }