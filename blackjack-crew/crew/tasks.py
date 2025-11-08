from crewai import Task, Agent
from game_core.game import BlackjackGame
from typing import Dict, List

def create_player_turn_tasks(player_name: str, agent: Agent, game: BlackjackGame) -> Task:
    """Creates a task for a single player's turn."""
    
    prompt = (
        f"It is {player_name}'s turn. Your goal is to decide whether to 'Hit' or 'Stand'.\n"
        f"1. Check the current state using game.get_player_state('{player_name}').\n"
        f"2. Based on your role and the state, decide 'Hit' or 'Stand'.\n"
        f"3. Call the Game Action Tool: GameActionTool.execute(player_name='{player_name}', action='[Your Choice]')\n"
        f"4. Review the tool's output. If the output contains 'TASK COMPLETE' (meaning BUST or STAND), then you are done.\n"
        f"5. If you chose 'Hit' and the output does NOT contain 'TASK COMPLETE', you MUST repeat steps 2 and 3 for the new state until your turn is over.\n"
        f"Initial State: {game.get_player_state(player_name)}\n"
    )

    # Human input is injected here to allow the user to interrupt and decide
    human_in = (player_name == 'Human')

    return Task(
        description=prompt,
        agent=agent,
        expected_output=f"A final summary of {player_name}'s turn, confirming their final hand and status (STAND or BUST).",
        human_input=human_in
    )

def get_all_tasks(agents: Dict[str, Agent], game: BlackjackGame) -> List[Task]:
    """Assembles all tasks in sequential order for the game flow."""
    
    # 1. Player Turns (Run each player's task sequentially)
    PLAYER_NAMES = ['Human', 'AI_Strategist', 'AI_Cautious']
    player_tasks = [
        create_player_turn_tasks(name, agents[name], game)
        for name in PLAYER_NAMES
    ]

    # 2. Dealer Turn
    dealer_turn_task = Task(
        description=(
            "All players have finished their turns. Execute the Dealer's turn by calling the Game Action Tool with action='Hit' repeatedly until the score is 17 or more, or bust.\n"
            "Start by checking the dealer's score. If it's 17 or more, call action='Stand'.\n"
            f"Dealer's current state:\n---\n{game.get_player_state('Dealer')}\n---"
        ),
        agent=agents['Dealer'],
        expected_output="A full log of all cards drawn by the Dealer and the final score, ending with STAND or BUST.",
    )

    # 3. Result Task (Reporting)
    result_task = Task(
        description="The entire round is over. Call the game.determine_winner() method to generate a final scorecard showing each player's score and the outcome (Win, Loss, or Push).",
        agent=agents['Dealer'],
        expected_output="A final scorecard showing the results for all players against the dealer.",
    )

    # Combine all tasks in sequence
    return player_tasks + [dealer_turn_task, result_task]