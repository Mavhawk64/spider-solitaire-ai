from bcolors import *
class Card(object):
    """Represents a standard playing card."""
    def __init__(self, suit=0, rank=2):
        self.suit = suit
        self.rank = rank
        self.face_up = False
    suit_names = ['Clubs', 'Diamonds', 'Spades', 'Hearts']
    suit_symbs = [bcolors.OKGREEN + '♣' + bcolors.ENDC, bcolors.HEADER + '♦' + bcolors.ENDC,bcolors.OKCYAN + '♠' + bcolors.ENDC, bcolors.FAIL + '♥' + bcolors.ENDC]
    rank_names = [None, 'Ace', '2', '3', '4', '5', '6', '7', 
        '8', '9', '10', 'Jack', 'Queen', 'King']
    def display(self):
        if not self.face_up or self.rank == 0:
            return '[ ]'
        if self.rank == 10:
            return self.rank_names[self.rank] + self.suit_symbs[self.suit]
        return ' ' + self.rank_names[self.rank][0] + self.suit_symbs[self.suit]
    def __str__(self):
        return '%s of %s' % (Card.rank_names[self.rank],
                             Card.suit_names[self.suit])
    def __eq__(self, __value: object) -> bool:
        return self.suit == __value.suit and self.rank == __value.rank
    def __lt__(self, __value: object) -> bool:
        if self.suit < __value.suit:
            return True
        elif self.suit == __value.suit:
            return self.rank < __value.rank
        else:
            return False
    def __gt__(self, __value: object) -> bool:
        if self.suit > __value.suit:
            return True
        elif self.suit == __value.suit:
            return self.rank > __value.rank
        else:
            return False
    def __le__(self, __value: object) -> bool:
        if self.suit < __value.suit:
            return True
        elif self.suit == __value.suit:
            return self.rank <= __value.rank
        else:
            return False
    def __ge__(self, __value: object) -> bool:
        if self.suit > __value.suit:
            return True
        elif self.suit == __value.suit:
            return self.rank >= __value.rank
        else:
            return False
    def __ne__(self, __value: object) -> bool:
        return self.suit != __value.suit or self.rank != __value.rank