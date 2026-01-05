from crewai.tools import BaseTool 
from pydantic import BaseModel, Field
from game_core.game import BlackjackGame

class GameToolInput(BaseModel):
    """Args for GameActionTool."""
    player_name: str = Field(description="Player name (e.g., 'Human', 'AI_1').")
    action: str = Field(description="'Hit' or 'Stand'.")

class GameActionTool(BaseTool):
    """Execute 'hit' or 'stand' for a named player.

    Accepts plain text ('hit'/'stand') or a JSON string {"action": "Hit"}.
    """
    name: str = "Blackjack Game Action Tool"
    description: str = "Make a move: Hit or Stand."
    args_schema: BaseModel = GameToolInput
    game: BlackjackGame = Field(default=None)

    def _run(self, player_name: str, action: str) -> str:
        if isinstance(action, str):
            text = action.strip()
            if text.startswith('{') and text.endswith('}'):
                try:
                    import json
                    payload = json.loads(text)
                    action_val = payload.get('action')
                    if isinstance(action_val, str):
                        text = action_val.strip()
                except Exception:
                    pass

            action_norm = text.lower()
            if action_norm in ('hit', 'hit()'):
                return self.game.hit(player_name)
            if action_norm in ('stand', 'stand()'):
                return self.game.stand(player_name)

        return "Invalid action. You must choose 'Hit' or 'Stand'."

    def execute_from_agent_output(self, player_name: str, output_text: str) -> str:
        """Parse agent output (JSON or text) and execute the corresponding action."""
        return self._run(player_name, output_text)


class ResultTool(BaseTool):
    """Return the deterministic final scorecard as JSON."""
    name: str = "Blackjack Result Tool"
    description: str = "Get final scorecard via game.determine_winner()."
    game: BlackjackGame = Field(default=None)

    def _run(self) -> str:
        try:
            import json
            result_text = self.game.determine_winner()
            return json.dumps({"scorecard": result_text})
        except Exception as e:
            return f"ERROR generating scorecard: {e}"
