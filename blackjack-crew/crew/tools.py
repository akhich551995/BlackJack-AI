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
        """Executes the chosen action in the game."""
        action = action.strip().lower()

        if action == 'hit':
            # We call the game's deterministic logic
            return self.game.hit(player_name)
        elif action == 'stand':
            # We call the game's deterministic logic
            return self.game.stand(player_name)
        
        return "Invalid action. You must choose 'Hit' or 'Stand'."
