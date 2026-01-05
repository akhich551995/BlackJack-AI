from crewai import Task, Agent
from game_core.game import BlackjackGame
from typing import Dict, List
import os

def create_player_turn_tasks(player_name: str, agent: Agent, game: BlackjackGame) -> Task:
    """Task prompting an AI to return a single JSON action: {"action":"Hit"|"Stand"}.

    The task includes the agent and the player's initial state.
    """

    prompt = (
        f"It is {player_name}'s turn. Decide: 'Hit' or 'Stand'.\n"
        f"Respond ONLY with JSON exactly like: {{\"action\": \"Hit\"}} or {{\"action\": \"Stand\"}}.\n"
        f"Initial State: {game.get_player_state(player_name)}\n"
    )

    return Task(
        description=prompt,
        agent=agent,
        expected_output=f"Final summary of {player_name}'s turn (final hand and status).",
        human_input=False
    )

def get_all_tasks(agents: Dict[str, Agent], game: BlackjackGame) -> List[Task]:
    """Return tasks for AI players only; dealer/results are handled in main."""

    player_tasks = [create_player_turn_tasks(name, agents[name], game) for name in agents.keys()]
    return player_tasks