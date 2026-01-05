import os
from pathlib import Path
from dotenv import load_dotenv

# Import components from the new structure
from game_core.game import BlackjackGame

# Player lists are created interactively at runtime

def run_blackjack_crew():
    """Start a console Blackjack round with Crew-managed AI players.

    Loads .env, configures model env vars, prompts for players, runs human
    turns synchronously, runs AI turns via Crew, executes dealer logic, and
    prints a deterministic final scorecard.
    """

    env_path = Path(__file__).resolve().parent / '.env'
    load_dotenv(dotenv_path=env_path)
    os.environ["CREWAI_LLM_PROVIDER"] = os.environ.get("CREWAI_LLM_PROVIDER", "openai")
    LLM_MODEL = os.environ.get("CREWAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    os.environ["CREWAI_MODEL"] = LLM_MODEL
    os.environ["MODEL"] = f"openai/{LLM_MODEL}"
    os.environ["MODEL_NAME"] = f"openai/{LLM_MODEL}"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        print("OPENAI_API_KEY found in environment.")
    else:
        print(f"WARNING: OPENAI_API_KEY not set. Checked {env_path}")

    # Import tools and Crew runtime (agents/tasks imported later)
    from crew.tools import GameActionTool
    from crewai import Crew, Process

    # --- 2. Ask user for table configuration ---
    # Prompt for total players and how many humans (interactive)
    def _ask_int(prompt: str, min_v: int, max_v: int) -> int:
        while True:
            try:
                v = int(input(f"{prompt} ({min_v}-{max_v}): ").strip())
                if min_v <= v <= max_v:
                    return v
            except Exception:
                pass
            print(f"Please enter an integer between {min_v} and {max_v}.")

    total_players = _ask_int("How many total players at the table", 1, 6)
    human_players = _ask_int("How many human players", 0, total_players)

    # Build player name lists: humans first, then AIs
    human_names = [f"Human{i+1}" for i in range(human_players)]
    ai_count = total_players - human_players
    ai_names = [f"AI_{i+1}" for i in range(ai_count)]
    player_names = human_names + ai_names

    # Initialize game with the chosen players and deal initial cards
    game = BlackjackGame(player_names=player_names)
    game.deal_initial_cards()

    # Initialize GameActionTool and a lightweight watchdog to limit repeated actions
    from collections import defaultdict

    class SafeGameActionTool(GameActionTool):
        """Tool wrapper that enforces a per-player action limit.

        Uses object.__setattr__/__getattribute__ to avoid pydantic/BaseModel
        attribute errors when adding runtime-only fields.
        """

        def __init__(self, game, max_actions_per_player: int = 10):
            try:
                super().__init__(game=game)
            except Exception:
                object.__setattr__(self, 'game', game)

            object.__setattr__(self, '_counts', defaultdict(int))
            object.__setattr__(self, 'max_actions', int(max_actions_per_player))

        def _run(self, player_name: str, action: str) -> str:
            counts = object.__getattribute__(self, '_counts')
            counts[player_name] += 1
            if counts[player_name] > object.__getattribute__(self, 'max_actions'):
                try:
                    if hasattr(self, 'game') and getattr(self, 'game') is not None:
                        try:
                            self.game.stand(player_name)
                        except Exception:
                            try:
                                self.game.active_players.discard(player_name)
                            except Exception:
                                pass
                except Exception:
                    pass
                return f"FORCE: max actions exceeded ({object.__getattribute__(self, 'max_actions')}). Forcing STAND. **TASK COMPLETE.**"

            return super()._run(player_name, action)

    max_actions = int(os.environ.get('CREWAI_MAX_ACTIONS', '10'))
    game_tool = SafeGameActionTool(game=game, max_actions_per_player=max_actions)

    def run_human_turn(game: BlackjackGame, tool: GameActionTool, player: str):
        """Blocking loop for a single human player's turn.

        Prompts for 'hit' or 'stand', updates the game via the tool, and
        prints the updated hand after each action.
        """
        print(f"\n--- {player} TURN ---")
        print(game.get_player_state(player))

        while player in game.active_players:
            hand = [str(c) for c in game.hands[player]]
            score = game.score_hand(game.hands[player])
            print(f"Your Hand: {hand} | Score: {score}")
            print(f"Dealer Up Card: {str(game.hands['Dealer'][0])}")

            choice = input("Choose action (hit/stand): ").strip().lower()
            if choice not in ('hit', 'stand'):
                print("Invalid input. Please type 'hit' or 'stand' (without quotes).")
                continue

            result = tool._run(player, choice)
            print(result)

            hand = [str(c) for c in game.hands[player]]
            score = game.score_hand(game.hands[player])
            print(f"Updated Hand: {hand} | Score: {score}")

            if "TASK COMPLETE" in str(result) or player not in game.active_players:
                print("Human turn complete.\n")
                break

    # Enforce human input (must be present); block until each human finishes their turn
    for human in human_names:
        if human in game.active_players:
            run_human_turn(game, game_tool, human)

    # Create AI agents and tasks
    from crew.agents import get_agents
    from crew.tasks import get_all_tasks
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

    # Create agents for AI players (llm_instance may be None)
    agents = get_agents(game_tool, ai_names, llm_instance)
    game_tasks = get_all_tasks(agents=agents, game=game)

    print("--- Initial Deal Complete ---")
    print(f"Dealer's Up Card: {str(game.hands['Dealer'][0])}")
    print("\nStarting Player Turns...")

    # Run AI agents with Crew sequentially
    all_agents = list(agents.values()) 

    # Instantiate Crew without internal verbose printing so our own
    # final scoreboard appears last in the CLI output.
    blackjack_crew = Crew(
        agents=all_agents,
        tasks=game_tasks,
        process=Process.sequential,
        verbose=False,
    )
    
    final_result = blackjack_crew.kickoff()
    # After AI turns complete, run deterministic dealer logic and compute results
    print("=========================================")
    print("========== BLACKJACK ROUND COMPLETE ==========")
    print("=========================================")

    # Ensure dealer plays out its hand according to rules (if not already)
    try:
        dealer_log = game.dealer_play()
        print("\n--- Dealer Play Log ---\n")
        print(dealer_log)
    except Exception as e:
        print("Dealer play failed:", e)

    # Compute deterministic final scoreboard using the game logic
    try:
        scorecard_text = game.determine_winner()
        print("\nFINAL SCORECARD (deterministic):\n")
        print(scorecard_text)
    except Exception as e:
        print("Could not determine winner programmatically:", e)
        scorecard_text = None

    # Optionally summarize the deterministic scorecard using an LLM (presentation only)
    summary = None
    if scorecard_text:
        summary_prompt = (
            "You are an announcer. Summarize the following Blackjack scorecard in one concise paragraph:\n\n"
            + scorecard_text
        )

        # Try langchain ChatOpenAI first if we created llm_instance earlier
        try:
            if llm_instance is not None:
                # Try common LangChain interfaces
                try:
                    summary = llm_instance.predict(summary_prompt)
                except Exception:
                    try:
                        summary = llm_instance(summary_prompt)
                    except Exception:
                        # Give up on langchain-style llm_instance
                        summary = None

            # Fallback to openai SDK if available and llm_instance was not used
            if summary is None:
                try:
                    import openai
                    openai.api_key = os.getenv('OPENAI_API_KEY')
                    resp = openai.ChatCompletion.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": summary_prompt}],
                        temperature=float(os.environ.get('CREWAI_TEMPERATURE', '0')),
                    )
                    summary = resp['choices'][0]['message']['content']
                except Exception:
                    summary = None
        except Exception:
            summary = None

    if summary:
        print("\n--- ANNOUNCEMENT (LLM) ---\n")
        print(summary)
    print("=========================================")


if __name__ == '__main__':
    run_blackjack_crew()
