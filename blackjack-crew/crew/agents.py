import os
from crewai import Agent
from crew.tools import GameActionTool

def get_agents(game_tool: GameActionTool) -> dict:
    """Defines and returns all agents required for the Blackjack game."""
    
    # LLM Configuration is loaded from environment variables
    LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash") 

    # Define Agents
    dealer_agent = Agent(
        role='Blackjack Dealer',
        goal='Strictly follow house rules: manage the game flow, deal cards, and execute the Dealer’s turn (hit until 17 or more, then stand).',
        backstory='You are the professional casino Dealer. Your actions are procedural and fixed by the game rules. You only act when all players are done.',
        tools=[game_tool],
        llm=LLM_MODEL,
        verbose=True,
        allow_delegation=False
    )

    human_agent = Agent(
        role='Human Player',
        goal='Make a move (Hit or Stand) when prompted, based on human intuition and risk.',
        backstory='You represent the human user. Your decisions are flexible, but you MUST use the Game Action Tool to register your choice.',
        tools=[game_tool],
        llm=LLM_MODEL, 
        verbose=True,
        allow_delegation=False
    )

    ai_strategist_agent = Agent(
        role='AI Blackjack Strategist',
        goal='Determine the mathematically optimal move (Hit or Stand) based on the Basic Strategy chart for Blackjack.',
        backstory="You are a genius AI player who always chooses the mathematically optimal action to minimize the house edge.",
        tools=[game_tool],
        llm=LLM_MODEL,
        verbose=True,
        allow_delegation=False
    )

    ai_cautious_agent = Agent(
        role='AI Cautious Player',
        goal='Play cautiously: Stand on any score of 15 or higher. Hit only if your score is 14 or lower.',
        backstory="You are an AI who prioritizes safety over aggression, always trying to avoid a bust.",
        tools=[game_tool],
        llm=LLM_MODEL,
        verbose=True,
        allow_delegation=False
    )

    return {
        'Dealer': dealer_agent,
        'Human': human_agent,
        'AI_Strategist': ai_strategist_agent,
        'AI_Cautious': ai_cautious_agent
    }