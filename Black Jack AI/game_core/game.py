import random
from typing import List, Dict

class Card:
    """Represents a playing card."""
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        # Using simple notation, e.g., 'A♠'
        suit_symbols = {'Spades': '♠', 'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣'}
        return f"{self.rank}{suit_symbols.get(self.suit, '')}" 

    def get_value(self) -> int:
        """Determines the card value (11 for Ace, 10 for face cards)."""
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11 # Ace is initially 11
        else:
            return int(self.rank)

class Deck:
    """Represents a deck of cards."""
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    SUITS = ['Spades', 'Hearts', 'Diamonds', 'Clubs']

    def __init__(self, num_decks: int = 4):
        self.cards: List[Card] = []
        for _ in range(num_decks):
            for rank in self.RANKS:
                for suit in self.SUITS:
                    self.cards.append(Card(rank, suit))
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self) -> Card:
        if not self.cards:
            # Simple reshuffle logic
            self.__init__() 
        return self.cards.pop()

class BlackjackGame:
    """Manages the state and rules of the Blackjack game."""
    def __init__(self, player_names: List[str]):
        self.deck = Deck()
        self.hands: Dict[str, List[Card]] = {name: [] for name in player_names}
        self.hands['Dealer'] = []
        self.player_names = player_names
        self.dealer_name = 'Dealer'
        self.active_players = set(player_names)
        self.game_status = "READY"

    def score_hand(self, hand: List[Card]) -> int:
        """Calculates the Blackjack score, handling Aces."""
        score = sum(card.get_value() for card in hand)
        num_aces = sum(1 for card in hand if card.rank == 'A')

        # Adjust for Aces
        while score > 21 and num_aces > 0:
            score -= 10  # Change Ace from 11 to 1
            num_aces -= 1
        return score

    def deal_initial_cards(self):
        """Deals 2 cards to each player and the dealer."""
        # Reset hands and active players for a new game
        self.hands = {name: [] for name in self.player_names}
        self.hands['Dealer'] = []
        self.active_players = set(self.player_names)

        for _ in range(2):
            for name in self.player_names + [self.dealer_name]:
                self.hands[name].append(self.deck.deal_card())
        self.game_status = "IN_PROGRESS"

    def get_player_state(self, player_name: str) -> str:
        """Returns the state information for an agent to make a decision."""
        hand = self.hands.get(player_name, [])
        score = self.score_hand(hand)

        # Dealer only shows the first card to players
        dealer_up_card = str(self.hands.get(self.dealer_name, ['None'])[0]) 
        
        status_message = "MUST CHOOSE HIT or STAND"
        if score > 21:
            status_message = "BUSTED"
        elif player_name not in self.active_players:
            status_message = "STAND (Turn Over)"

        state_summary = (
            f"Your Name: {player_name}\n"
            f"Your Current Hand: {[str(c) for c in hand]}\n"
            f"Your Current Score: {score}\n"
            f"Dealer's Up Card: {dealer_up_card}\n"
            f"Current Status: {status_message}"
        )
        return state_summary

    def hit(self, player_name: str) -> str:
        """Adds a card to a player's hand."""
        if player_name not in self.active_players:
            return f"{player_name} is already finished (stood or busted). No action taken."
        
        card = self.deck.deal_card()
        self.hands[player_name].append(card)
        score = self.score_hand(self.hands[player_name])

        if score > 21:
            self.active_players.discard(player_name)
            return f"{player_name} draws {card}. Score is {score}. BUST! Player is out. **TASK COMPLETE.**"
        elif score == 21:
            self.active_players.discard(player_name)
            return f"{player_name} draws {card}. Score is 21. STAND automatically. **TASK COMPLETE.**"
        else:
            return f"{player_name} draws {card}. New score is {score}. You must call the tool again to decide: HIT or STAND."

    def stand(self, player_name: str) -> str:
        """Sets a player's status to stand."""
        if player_name in self.active_players:
            score = self.score_hand(self.hands[player_name])
            self.active_players.discard(player_name)
            return f"{player_name} STANDS with a final score of {score}. Turn over. **TASK COMPLETE.**"
        return f"{player_name} has already finished their turn. No action taken."

    def dealer_play(self) -> str:
        """Dealer hits until score is 17 or more."""
        log = []
        hand = self.hands[self.dealer_name]
        dealer_score = self.score_hand(hand)
        log.append(f"Dealer reveals full hand: {[str(c) for c in hand]}, Initial Score: {dealer_score}")

        while dealer_score < 17:
            card = self.deck.deal_card()
            hand.append(card)
            dealer_score = self.score_hand(hand)
            log.append(f"Dealer hits, draws {card}. New score: {dealer_score}")

        if dealer_score > 21:
            log.append(f"Dealer BUSTS with a score of {dealer_score}.")
        else:
            log.append(f"Dealer STANDS with a score of {dealer_score}.")
        
        self.game_status = "FINISHED"
        return "\n".join(log)

    def determine_winner(self) -> str:
        """Compares scores and determines outcomes."""
        dealer_score = self.score_hand(self.hands[self.dealer_name])
        dealer_busted = dealer_score > 21
        results = [f"\n--- GAME RESULTS ---\nDealer Final Hand: {[str(c) for c in self.hands[self.dealer_name]]}, Score: {dealer_score} ({'BUSTED' if dealer_busted else 'STAND'})"]

        for name, hand in self.hands.items():
            if name == self.dealer_name:
                continue

            player_score = self.score_hand(hand)
            player_hand = [str(c) for c in hand]
            
            outcome = ""
            if player_score > 21:
                outcome = "BUSTED - Dealer WINS."
            elif dealer_busted:
                outcome = "Dealer BUSTED - Player WINS."
            elif player_score > dealer_score:
                outcome = "Player WINS."
            elif player_score < dealer_score:
                outcome = "Dealer WINS."
            else:
                outcome = "PUSH (Tie)."
            
            results.append(f"{name} | Score: {player_score} | Hand: {player_hand} | Outcome: {outcome}")
        
        return "\n".join(results)

# Ensure game_core is treated as a package