from crewai.tools import BaseTool 
from pydantic import BaseModel, Field
from game_core.game import BlackjackGame

class GameToolInput(BaseModel):
    """Pydantic model for the tool input."""
    player_name: str = Field(description="The name of the player making the move (e.g., 'Human', 'AI_Strategist').")
    action: str = Field(description="The action to take: 'Hit' to take a card, or 'Stand' to finish the turn.")

class GameActionTool(BaseTool):
    """
    A tool for players to interact with the Blackjack game state.
    It executes a 'Hit' or 'Stand' action for the current player.
    """
    name: str = "Blackjack Game Action Tool"
    description: str = "Use this tool to make your move (Hit or Stand). Input format is 'player_name, action'."
    args_schema: BaseModel = GameToolInput
    game: BlackjackGame = Field(default=None) # Holds the game instance

    def _run(self, player_name: str, action: str) -> str:
        """Executes the chosen action in the game.

        The method accepts either a plain action string ('hit'/'stand') or a
        JSON string like '{"action": "Hit"}'. It is robust to whitespace
        and capitalization.
        """
        if isinstance(action, str):
            text = action.strip()
            # Allow agents to return a JSON object as text: {"action": "Hit"}
            if text.startswith('{') and text.endswith('}'):
                try:
                    import json
                    payload = json.loads(text)
                    action_val = payload.get('action')
                    if isinstance(action_val, str):
                        text = action_val.strip()
                except Exception:
                    # fall through to plain parsing
                    pass

            action_norm = text.lower()
            if action_norm == 'hit' or action_norm == 'hit()':
                return self.game.hit(player_name)
            if action_norm == 'stand' or action_norm == 'stand()':
                return self.game.stand(player_name)

        return "Invalid action. You must choose 'Hit' or 'Stand'."

    def execute_from_agent_output(self, player_name: str, output_text: str) -> str:
        """Helper to parse free-form agent output and execute the corresponding action.

        This attempts JSON parsing first, then falls back to scanning for the keywords
        'hit' or 'stand' in the output. Returns the tool result string.
        """
        # Try to reuse _run's parsing logic
        return self._run(player_name, output_text)


class ResultTool(BaseTool):
    """Tool to expose the game's final scorecard deterministically.

    Returns a JSON string containing the final results produced by
    BlackjackGame.determine_winner(). Agents can call this tool to get a
    machine-readable scoreboard instead of computing it themselves.
    """
    name: str = "Blackjack Result Tool"
    description: str = "Returns the final scorecard as JSON by calling game.determine_winner()."
    game: BlackjackGame = Field(default=None)

    def _run(self) -> str:
        try:
            import json
            result_text = self.game.determine_winner()
            # Return structured JSON with the raw text included for presentation
            return json.dumps({"scorecard": result_text})
        except Exception as e:
            return f"ERROR generating scorecard: {e}"
