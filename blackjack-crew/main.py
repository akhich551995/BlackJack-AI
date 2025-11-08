import os
from dotenv import load_dotenv
from crewai import Crew, Process

# Import components from the new structure
from game_core.game import BlackjackGame
from crew.agents import get_agents
from crew.tasks import get_all_tasks
from crew.tools import GameActionTool

# Define the player list globally for setup
PLAYER_NAMES = ['Human', 'AI_Strategist', 'AI_Cautious']

def run_blackjack_crew():
    """Initializes environment, game, crew, and runs the simulation."""
    
    # --- 1. Load Environment Variables ---
    # This step loads GEMINI_API_KEY, LLM_MODEL, etc. from the .env file
    load_dotenv()

    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file.")
        return

    print("=========================================")
    print("======== STARTING BLACKJACK CREW ========")
    print("=========================================")
    
    # --- 2. Initialize Game State and Tool ---
    game = BlackjackGame(player_names=PLAYER_NAMES)
    game.deal_initial_cards() # Deal the first cards before starting the crew

    # The tool must be initialized with the active game instance
    game_tool = GameActionTool(game=game)

    # --- 3. Initialize Agents and Tasks ---
    agents = get_agents(game_tool=game_tool)
    game_tasks = get_all_tasks(agents=agents, game=game)

    print("--- Initial Deal Complete ---")
    print(f"Dealer's Up Card: {str(game.hands['Dealer'][0])}")
    print("\nStarting Player Turns...")

    # --- 4. Create and Run Crew ---
    
    # Pass all agent instances (Dealer + Players) to the Crew
    all_agents = list(agents.values()) 

    blackjack_crew = Crew(
        agents=all_agents,
        tasks=game_tasks,
        process=Process.sequential, 
        verbose=True,
    )
    
    final_result = blackjack_crew.kickoff()

    print("=========================================")
    print("========== BLACKJACK GAME OVER ==========")
    print("=========================================")
    print("\nFINAL CREW OUTPUT (The result task):\n")
    print(final_result)
    print("=========================================")


if __name__ == '__main__':
    run_blackjack_crew()