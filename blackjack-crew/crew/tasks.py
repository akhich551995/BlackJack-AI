from crewai import Task, Agent
from game_core.game import BlackjackGame
from typing import Dict, List
import os

def create_player_turn_tasks(player_name: str, agent: Agent, game: BlackjackGame) -> Task:
    """Creates a task for a single player's turn."""
    
    # Initialize human_input flag (default is False)
    human_in = False

    # If AUTO_ACCEPT_HUMAN is set to 'true' in the environment, skip human prompts
    auto_accept = os.getenv('AUTO_ACCEPT_HUMAN', 'false').lower() in ('1', 'true', 'yes')

    if player_name == 'Human':
        # Custom prompt for the human player to guide user input
        prompt = (
            f"It is the **HUMAN PLAYER's** turn. Please follow these steps:\n"
            f"1. Review the current game state provided below.\n"
            f"2. Decide whether to **'Hit'** or **'Stand'**.\n"
            f"3. Your required output format is a single call to the Game Action Tool, e.g.: `GameActionTool.execute(player_name='Human', action='Hit')`.\n"
            f"4. If you choose 'Hit', the tool will return a new state. You MUST repeat this process for the new state until you 'Stand' or 'Bust'.\n"
            f"Initial State:\n{game.get_player_state(player_name)}\n"
            f"--------------------------------------------------\n"
            f"**HUMAN INPUT REQUIRED**: Provide the full tool execution call (e.g., GameActionTool.execute(...))."
        )
        # Explicitly set the flag to True for the Human player task unless
        # auto-accept is enabled, in which case the task will not require
        # interactive human input and the Crew flow completes non-interactively.
        human_in = not auto_accept
    else:
        # Standard prompt for AI agents
        # Enforce a structured JSON response to make parsing deterministic. ONLY
        # output a single JSON object with an 'action' field whose value is
        # either "Hit" or "Stand". Example: {"action": "Hit"}
        prompt = (
            f"It is {player_name}'s turn. Your goal is to decide whether to 'Hit' or 'Stand'.\n"
            f"1. Check the current state using game.get_player_state('{player_name}').\n"
            f"2. Decide the single action to take next.\n"
            f"3. IMPORTANT: Output ONLY a single JSON object exactly like: {{\"action\": \"Hit\"}} or {{\"action\": \"Stand\"}} and nothing else.\n"
            f"4. After the tool executes, inspect the result. If it contains 'TASK COMPLETE', your turn is over. If not, repeat this process for the new state.\n"
            f"Initial State: {game.get_player_state(player_name)}\n"
        )

    return Task(
        description=prompt,
        agent=agent,
        expected_output=f"A final summary of {player_name}'s turn, confirming their final hand and status (STAND or BUST).",
        # The Task uses the conditional human_in flag here
        human_input=human_in 
    )

def get_all_tasks(agents: Dict[str, Agent], game: BlackjackGame) -> List[Task]:
    """Assembles all tasks in sequential order for the game flow."""
    # 1. Player Turns (Run each AI player's task sequentially)
    # The agents dict contains only AI players; humans are handled in main.py.
    player_tasks = [
        create_player_turn_tasks(name, agents[name], game)
        for name in agents.keys()
    ]

    # 2. Dealer Turn
    # NOTE: Dealer and final result are handled procedurally in main.py for
    # deterministic behavior, so the Crew-managed tasks list contains only
    # AI player tasks. The main flow will run dealer_play() and determine_winner().
    return player_tasks