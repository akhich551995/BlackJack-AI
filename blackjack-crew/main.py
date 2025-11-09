import os
from pathlib import Path
from dotenv import load_dotenv

# Import components from the new structure
from game_core.game import BlackjackGame

# Define the player list globally for setup
PLAYER_NAMES = ['Human', 'AI_Strategist', 'AI_Cautious']

def run_blackjack_crew():
    """Initializes environment, game, crew, and runs the simulation."""
    
    # --- 1. Load Environment Variables ---
    # Load .env located next to this script (robust when working dir differs)
    env_path = Path(__file__).resolve().parent / '.env'
    load_dotenv(dotenv_path=env_path)
    # Get desired provider/model from env. Default to OpenAI provider.
    os.environ["CREWAI_LLM_PROVIDER"] = os.environ.get("CREWAI_LLM_PROVIDER", "openai")
    # CREWAI_MODEL is the model name CrewAI Agents will use when created from a string
    LLM_MODEL = os.environ.get("CREWAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    # Explicitly set CREWAI_MODEL and generic MODEL envs (direct assignment)
    os.environ["CREWAI_MODEL"] = LLM_MODEL
    os.environ["MODEL"] = f"openai/{LLM_MODEL}"
    os.environ["MODEL_NAME"] = f"openai/{LLM_MODEL}"

    # Do not require Google/Gemini keys anymore; expect OPENAI_API_KEY in .env
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        print("OPENAI_API_KEY found in environment.")
    else:
        print(f"WARNING: OPENAI_API_KEY not set. Checked {env_path}")

    print("=========================================")
    print(f"======== STARTING BLACKJACK CREW (Model: {LLM_MODEL}) ========")
    print("=========================================")
    # Import crew modules after environment is configured to avoid crewi
    # creating a fallback LLM from the environment during module import.
    # We only import the GameActionTool now because the Human turn is handled
    # synchronously below; agents/tasks are imported after the human turn so
    # the Human is not represented as a Crew-managed task.
    from crew.tools import GameActionTool
    # Import crewi runtime objects here (we will construct the Crew later).
    from crewai import Crew, Process

    # --- 2. Initialize Game State and Tool ---
    game = BlackjackGame(player_names=PLAYER_NAMES)
    game.deal_initial_cards() # Deal the first cards before starting the crew

    # The tool must be initialized with the active game instance
    # Wrap the real GameActionTool with a lightweight watchdog to prevent
    # infinite loops: limit the number of actions a single player can take
    # during their turn. The limit can be overridden with CREWAI_MAX_ACTIONS.
    from collections import defaultdict

    class SafeGameActionTool(GameActionTool):
        def __init__(self, game, max_actions_per_player: int = 10):
            # Some BaseTool implementations don't expect __init__, so we
            # set the game attribute directly and keep subclass behavior.
            try:
                super().__init__(game=game)
            except Exception:
                # Fallback: set attribute directly
                self.game = game
            self._counts = defaultdict(int)
            self.max_actions = int(max_actions_per_player)

        def _run(self, player_name: str, action: str) -> str:
            # Increment and check watchdog count before delegating
            self._counts[player_name] += 1
            if self._counts[player_name] > self.max_actions:
                # Force the player's turn to end to avoid infinite loops
                # Ensure the player is removed from active_players
                try:
                    self.game.active_players.discard(player_name)
                except Exception:
                    pass
                return f"FORCE: max actions exceeded ({self.max_actions}). Forcing STAND. **TASK COMPLETE.**"

            return super()._run(player_name, action)

    max_actions = int(os.environ.get('CREWAI_MAX_ACTIONS', '10'))
    game_tool = SafeGameActionTool(game=game, max_actions_per_player=max_actions)

    # --- Human turn: blocking CLI loop (synchronous) ---
    def run_human_turn(game: BlackjackGame, tool: GameActionTool):
        player = 'Human'
        # Show initial state
        print("\n--- HUMAN TURN ---")
        print(game.get_player_state(player))

        # Loop until player stands or busts
        while player in game.active_players:
            # Display concise info
            hand = [str(c) for c in game.hands[player]]
            score = game.score_hand(game.hands[player])
            print(f"Your Hand: {hand} | Score: {score}")
            print(f"Dealer Up Card: {str(game.hands['Dealer'][0])}")

            choice = input("Choose action (hit/stand): ").strip().lower()
            if choice not in ('hit', 'stand'):
                print("Invalid input. Please type 'hit' or 'stand' (without quotes).")
                continue

            # Call the GameActionTool directly using the underlying _run method
            result = tool._run(player, choice)
            print(result)

            # Show updated hand after action
            hand = [str(c) for c in game.hands[player]]
            score = game.score_hand(game.hands[player])
            print(f"Updated Hand: {hand} | Score: {score}")

            # If TASK COMPLETE is returned or player no longer active, break
            if "TASK COMPLETE" in str(result) or player not in game.active_players:
                print("Human turn complete.\n")
                break

    # Enforce human input (must be present); block until human finishes turn
    run_human_turn(game, game_tool)

    # --- 3. Initialize Agents and Tasks ---
    # Now import agents and tasks and build the Crew-managed tasks for AI players.
    from crew.agents import get_agents
    from crew.tasks import get_all_tasks

    # Option A: explicitly construct an LLM instance (preferred) so we can
    # control temperature and other params. We try to create a LangChain
    # ChatOpenAI instance if the package is available. If not, we fall back
    # to letting Crew construct the LLM from the CREWAI_MODEL env string.
    llm_instance = None
    desired_temp = float(os.environ.get('CREWAI_TEMPERATURE', '0'))
    try:
        from langchain.chat_models import ChatOpenAI
        # instantiate with explicit temperature to enforce deterministic outputs
        llm_instance = ChatOpenAI(model_name=LLM_MODEL, temperature=desired_temp)
        print(f"Using explicit ChatOpenAI LLM instance (temperature={desired_temp})")
    except Exception:
        # LangChain ChatOpenAI not available; fall back to env-driven model creation
        print("LangChain ChatOpenAI not available; falling back to CREW-managed model string.")

    # Let get_agents either receive an explicit llm_instance or a model string
    agents = get_agents(game_tool=game_tool, llm_instance=llm_instance)
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