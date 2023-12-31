import Card
import random

class Deck(object):
    """Represents a deck/set of cards."""
    def __init__(self, decks=1, suits=4, seed=None):
        random.seed(seed)
        self.cards = []
        for _ in range(decks):
            for suit in range(suits):
                for rank in range(1, 14):
                    self.cards.append(Card.Card(suit, rank))
    
    def __str__(self):
        res = []
        for card in self.cards:
            res.append(str(card))
        return '\n'.join(res)
    
    def __len__(self):
        return len(self.cards)
    
    def add_suit(self, suit):
        for rank in range(1, 14):
            self.cards.append(Card.Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.cards)

    def sort(self):
        self.cards.sort()
    
    def deal(self):
        """Deals and returns the top card from the deck."""
        return self.cards.pop() if self.cards else None